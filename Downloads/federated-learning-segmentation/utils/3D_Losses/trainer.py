import logging
import os
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter

from . import utils

from quantize.quantization_T import optimization_step as optimization_step_T, quantize
from for_data import ModelSaver
from vis_utils import colorEncode
import torchvision.utils as vutils

from scipy.stats import rice
from scipy.io import loadmat


class UNet3DTrainer:
    """3D UNet trainer.

    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        loss_criterion (callable): loss function
        accuracy_criterion (callable): used to compute training/validation accuracy (such as Dice or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        max_patience (int): number of validation runs with no improvement
            after which the training will be stopped
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        best_val_accuracy (float): best validation accuracy so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
    """

    def __init__(self, model, optimizer, loss_criterion, accuracy_criterion,
                 device, loaders, checkpoint_dir,
                 max_num_epochs=200, max_num_iterations=1e5, max_patience=20,
                 validate_after_iters=100, log_after_iters=50,  # 100
                 validate_iters=None, best_val_accuracy=float('-inf'),
                 num_iterations=0, num_epoch=1, logger=None, ternary_style=False,
                 ternary_optimizers=None, num_bits=32, shift_down=False,
                 HYPERPARAMETER_T=0.05, num_sf=0, activations='relu', betas=None,
                 ternary_net=False, ternary_ops=False, lr_for_log=0.0,
                 scale_by_alpha=False, loss_for_log='loss', trial=False,
                 train_also_biases=False, start_pretrained=False, dataset='hippo',
                 inputs_style=None, cluster=False):
        if logger is None:
            self.logger = utils.get_logger('UNet3DTrainer', level=logging.DEBUG)
        else:
            self.logger = logger

        self.logger.info(f"Sending the model to '{device}'")
        self.model = model.to(device)
        self.logger.debug(model)

        self.optimizer = optimizer
        self.loss_criterion = loss_criterion
        self.accuracy_criterion = accuracy_criterion
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.best_val_accuracy = best_val_accuracy
        # self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))

        self.num_iterations = num_iterations
        self.num_epoch = num_epoch
        # used for early stopping
        self.max_patience = max_patience
        self.patience = max_patience

        self.lr_for_log = lr_for_log
        self.loss_for_log = loss_for_log
        self.activations = activations
        self.ternary_style = ternary_style
        self.dataset = dataset
        if self.ternary_style:
            self.num_bits = num_bits
            self.shift_down = shift_down
            self.HYPERPARAMETER_T = HYPERPARAMETER_T
            self.num_sf = num_sf
            # self.t_tanh = t_tanh
            # self.activations = 't_tanh' if self.t_tanh else 'ReLU'
            self.ternary_ops = 'quant' if ternary_ops else 'std'
            self.ternary_net = ternary_net
            self.weights_kind = 'ternary' if num_bits == 1.5 else 'binary'
            self.approach = 'TernaryNet' if self.ternary_net else 'TTQ'
            self.scale_by_alpha = scale_by_alpha
            if not self.ternary_net:
                self.optimizer_fp = ternary_optimizers[0]
                self.optimizer_sf = ternary_optimizers[1]
                self.scaling_factors = self.optimizer_sf.param_groups[0]['params']
                if self.scale_by_alpha:
                    self.approach = self.approach + '+a'
                if train_also_biases:
                    self.approach = self.approach + '+b'
            else:
                self.scaling_factors = None
            if self.activations == 'ttanh':
                self.betas = betas
        else:
            self.num_bits = 32
            self.activations = 'relu'  # in any case
            # self.t_tanh = False
            self.ternary_net = False
            self.ternary_ops = 'std'
            self.approach = 'std'
            self.weights_kind = 'full'
            self.scaling_factors = None
        if self.checkpoint_dir == '.':
            tmp = 'my_out' if cluster else 'home/stefano'
            self.checkpoint_dir = f'/{tmp}/logs_{self.dataset}/{self.weights_kind}_{self.approach}_' \
                                  f'{self.activations}_{self.ternary_ops}_{self.lr_for_log}_' \
                                  f'{self.loss_for_log}'
            if start_pretrained:
                self.checkpoint_dir = self.checkpoint_dir + '_pretr'
            if inputs_style is not None:
                self.checkpoint_dir = self.checkpoint_dir + '_' + inputs_style
            if trial:
                self.checkpoint_dir = self.checkpoint_dir + f'_trial_{time.time() // 1}'
            self.checkpoint_dir = self.checkpoint_dir + f'_{self.dataset}'
            self.logger.info(f'checkpoint_dir updated to {self.checkpoint_dir}')
        if self.ternary_style and not self.ternary_net and self.num_bits in [1, 1.5]:
            self.model_saver = ModelSaver(self.optimizer, self.optimizer_sf, self.model,
                                          self.checkpoint_dir, num_bits=self.num_bits, num_sf=self.num_sf)
        self.writer = SummaryWriter(log_dir=self.checkpoint_dir)
        self.colors = loadmat('resources/color150.mat')['colors']
        self.colors[0] = [0, 0, 0]

    @classmethod
    def from_checkpoint(cls, checkpoint_path, model, optimizer, loss_criterion, accuracy_criterion, loaders,
                        device, logger=None, also_compressed=False, ternary_style=False,
                        ternary_optimizers=None, num_bits=32, shift_down=False,
                        HYPERPARAMETER_T=0.05, num_sf=0, activations='relu', betas=None,
                        ternary_net=False, ternary_ops=False, lr_for_log=0.0,
                        scale_by_alpha=False, loss_for_log='loss', trial=False,
                        train_also_biases=False, dataset='hippo', inputs_style=None, cluster=False,
                        new_dir=False):
        logger.info(f"Loading checkpoint '{checkpoint_path}'...")

        if ternary_style:
            state = utils.load_checkpoint(checkpoint_path, model, optimizer,
                                          ModelSaver.decompress(torch.load(checkpoint_path + 'compressed_best.pytorch',
                                                                map_location=device), num_bits=num_bits,
                                                                shift_down=shift_down, num_sf=num_sf))
        else:
            state = utils.load_checkpoint(checkpoint_path, model, optimizer, load_best=False)

        logger.info(
            f"Checkpoint loaded. Epoch: {state['epoch']}. "
            f"Best val accuracy: {state['best_val_accuracy']}. "
            f"Num_iterations: {state['num_iterations']}")
        checkpoint_dir = os.path.split(checkpoint_path)[0] if not new_dir else '.'
        return cls(model, optimizer, loss_criterion, accuracy_criterion, torch.device(state['device']), loaders,
                   checkpoint_dir,
                   best_val_accuracy=state['best_val_accuracy'],
                   num_iterations=state['num_iterations'],
                   num_epoch=state['epoch'],
                   max_num_epochs=state['max_num_epochs'],
                   max_num_iterations=state['max_num_iterations'],
                   max_patience=state['max_patience'],
                   validate_after_iters=state['validate_after_iters'],
                   log_after_iters=state['log_after_iters'],
                   validate_iters=state['validate_iters'],
                   logger=logger, ternary_style=ternary_style,
                   ternary_optimizers=ternary_optimizers, num_bits=num_bits, shift_down=shift_down,
                   HYPERPARAMETER_T=HYPERPARAMETER_T, num_sf=num_sf, activations=activations, betas=betas,
                   ternary_net=ternary_net, ternary_ops=ternary_ops, lr_for_log=lr_for_log,
                   scale_by_alpha=scale_by_alpha, loss_for_log=loss_for_log, trial=trial,
                   train_also_biases=train_also_biases, dataset=dataset, inputs_style=inputs_style, cluster=cluster)

    def fit(self):
        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train(self.loaders['train'], self.ternary_style and self.ternary_net)

            if should_terminate:
                break

            self.num_epoch += 1

    def train(self, train_loader, ternary_net=False):
        """Trains the model for 1 epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): training data loader

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        start_time = time.time()

        train_losses = utils.RunningAverage()
        train_accuracy = utils.RunningAverage()

        # val_accuracy, val_time = self.validate(self.loaders['val'])
        # self.logger.info(f'{val_accuracy}')

        # sets the model in training mode
        self.model.train()

        if self.ternary_style and self.num_bits == 1.5 and self.activations == 'ttanh':  # self.t_tanh:
            new_beta = float(self.betas[self.num_epoch])
            self.model.set_beta(new_beta)
            self.logger.info(f'Updated beta to {new_beta}')

        for i, t in enumerate(train_loader):
            if len(t) == 3:
                input, target, b = t
                input = input.unsqueeze(1).float()
                input, target = input.to(self.device), target.to(self.device)
                weight = None
            else:
                input, target, b, weight = t
                input, target, weight = input.to(self.device), target.to(self.device), weight.to(self.device)

            if hasattr(self.loss_criterion, 'ignore_index') and self.loss_criterion.ignore_index is not None:
                unique_labels = torch.unique(target)
                if len(unique_labels) == 1 and unique_labels.item() == self.loss_criterion.ignore_index:
                    self.logger.info(f'Skipping training batch {i} (contains only ignore_index)...')
                    continue

            # target = target.squeeze(1)

            if ternary_net:
                '''stored_weights = [
                    p for n, p in self.model.named_parameters()
                    if 'conv' in n and 'weight' in n and 'norm' not in n and n.requires_grad
                ]'''
                stored_weights = []
                for n, p in self.model.named_parameters():
                    if 'conv' in n and 'weight' in n and 'norm' not in n and p.requires_grad:
                        stored_weights.append(p.data)
                        p.data = quantize(p.data, None, num_bits=self.num_bits, ternary_net=True,
                                          shift_down=self.shift_down)

            output, loss, accuracy = self._forward_pass(input, target, weight)

            train_losses.update(loss.item(), input.size(0))
            train_accuracy.update(accuracy.item(), input.size(0))

            # compute gradients and update parameters
            if self.ternary_style and not self.ternary_net:
                self.optimizer_fp.zero_grad()
                self.optimizer_sf.zero_grad()
            self.optimizer.zero_grad()

            loss.backward()

            if ternary_net:
                i = 0
                for n, p in self.model.named_parameters():
                    if 'conv' in n and 'weight' in n and 'norm' not in n and p.requires_grad:
                        p.data = stored_weights[i]
                        i += 1

            if not self.ternary_style or (self.ternary_style and self.ternary_net):
                self.optimizer.step()
            else:
                optimization_step_T([self.optimizer, self.optimizer_fp, self.optimizer_sf],
                                    self.device, t=self.HYPERPARAMETER_T, num_bits=self.num_bits,
                                    shift_down=self.shift_down, num_sf=self.num_sf, scale_by_alpha=self.scale_by_alpha)

            if self.num_iterations % self.log_after_iters == 0:
                self.logger.info(
                    f'Training iteration {self.num_iterations}. Batch {i}. '
                    f'Epoch [{self.num_epoch}/{self.max_num_epochs}]')
                self.logger.info(
                    f'Training stats. Loss: {train_losses.avg}. Accuracy: {train_accuracy.avg}')
                # self.logger.info(f'GPU Memory usage: {torch.cuda.memory_allocated()}')
            self.num_iterations += 1

        end_time = time.time()
        self.logger.info(f'Time elapsed for this epoch: {end_time - start_time} s')

        # if self.num_iterations % self.log_after_iters == 0:
        # log stats, params and images
        if self.scaling_factors is None:
            self._log_stats('train', train_losses.avg, train_accuracy.avg, end_time - start_time)
        else:
            self._log_stats('train', train_losses.avg, train_accuracy.avg, end_time - start_time,
                            avg_sf=torch.mean(torch.stack(self.scaling_factors)))
        self._log_params()

        # normalize output (during training the network outputs logits only)
        # output = self.model.final_activation(output)
        self._log_images(input, target, output)

        if ternary_net:
            '''stored_weights = [
                p for n, p in self.model.named_parameters()
                if 'conv' in n and 'weight' in n and 'norm' not in n and n.requires_grad
            ]'''
            stored_weights = []
            for n, p in self.model.named_parameters():
                if 'conv' in n and 'weight' in n and 'norm' not in n and p.requires_grad:
                    stored_weights.append(p.data)
                    p.data = quantize(p.data, None, num_bits=self.num_bits, ternary_net=True,
                                      shift_down=self.shift_down)

        # if self.num_iterations % self.validate_after_iters == 0:  # removed to validate after each epoch
        # evaluate on validation set
        val_accuracy, val_time = self.validate(self.loaders['val'])

        if ternary_net:
            i = 0
            for n, p in self.model.named_parameters():
                if 'conv' in n and 'weight' in n and 'norm' not in n and p.requires_grad:
                    p.data = stored_weights[i]
                    i += 1

        self.logger.info(f'{val_accuracy} {val_time} {self.num_epoch} {self.weights_kind} '
                         f'{self.approach} {self.activations} {self.ternary_ops} {self.lr_for_log} '
                         f'{self.loss_for_log} {self.checkpoint_dir}')


        # remember best validation metric
        is_best = self._is_best_val_accuracy(val_accuracy)

        # save checkpoint
        self._save_checkpoint(is_best)
        if self.ternary_style and not self.ternary_net and self.num_bits in [1, 1.5]:
            self.model_saver.compress_model(is_best)
            self.logger.info(f'saved compress model')

        if self._check_early_stopping(is_best):
            self.logger.info(
                f'Validation accuracy did not improve for the last '
                f'{self.max_patience} validation runs. Early stopping...')
            return True

        if self.max_num_iterations < self.num_iterations:
            self.logger.info(
                f'Maximum number of iterations {self.max_num_iterations} exceeded. Finishing training...')
            return True

        return False

    def validate(self, val_loader, with_noise=False):
        self.logger.info('Validating...')

        val_losses = utils.RunningAverage()
        val_accuracy = utils.RunningAverage()

        try:
            with torch.no_grad():
                start_time = time.time()

                if self.dataset == 'malc':
                    dice_coeffs = torch.zeros((15, 28)).to(self.device)
                    count_b = np.zeros(15)

                for i, t in enumerate(val_loader):
                    if len(t) == 3:
                        input, target, b = t
                        input = input.unsqueeze(1).float()
                        input, target = input.to(self.device), target.to(self.device)
                        weight = None
                    else:
                        input, target, b, weight = t
                        input, target, weight = input.to(self.device), target.to(self.device), weight.to(self.device)

                    if hasattr(self.loss_criterion, 'ignore_index') and self.loss_criterion.ignore_index is not None:
                        unique_labels = torch.unique(target)
                        if len(unique_labels) == 1 and unique_labels.item() == self.loss_criterion.ignore_index:
                            self.logger.info(f'Skipping validation batch {i} (contains only ignore_index)...')
                            continue

                    target = target.squeeze(1)

                    if with_noise:
                        input = input + torch.from_numpy(0.05 *
                                                         rice.rvs(0.775, size=(input.shape[0], input.shape[1],
                                                                              input.shape[2], input.shape[3],
                                                                              input.shape[4]))).float().to(self.device)

                    output, loss, accuracy = self._forward_pass(input, target, weight,
                                                                is_training=(False if self.dataset == 'malc' else True))

                    val_losses.update(loss.item(), input.size(0))
                    if self.dataset == 'hippo':
                        val_accuracy.update(accuracy.item(), input.size(0))
                    else:
                        dice_coeffs[b.long()] += accuracy
                        # print(accuracy)
                        count_b[b.long().detach().cpu().numpy()] += 1

                    if self.validate_iters is not None and self.validate_iters <= i:
                        # stop validation
                        break
                    if i % self.log_after_iters == 0:
                        self.logger.info(f'Validation iteration {i}')
                        # self.logger.info(f'GPU Memory usage: {torch.cuda.memory_allocated()}')

                if self.dataset == 'malc':
                    for j in range(dice_coeffs.shape[0]):
                        dice_coeffs[j] /= count_b[j]
                        val_accuracy.update(np.mean(dice_coeffs[j].detach().cpu().numpy()), j)

                end_time = time.time()
                self._log_stats('val', val_losses.avg, val_accuracy.avg, end_time-start_time)
                self.logger.info(f'Validation finished. Loss: {val_losses.avg}. Accuracy: {val_accuracy.avg}')
                self.logger.info(f'Time elapsed for this validation run: {end_time - start_time} s')
                return val_accuracy.avg, end_time-start_time
        finally:
            self.model.train()

    def _forward_pass(self, input, target, weight=None, is_training=True):
        # forward pass
        output = self.model(input)

        # compute the loss
        if weight is None:
            loss = self.loss_criterion(output, target)
        else:
            loss = self.loss_criterion(output, target, weight)

        # normalize logits and compute the accuracy criterion
        accuracy = self.accuracy_criterion(self.model.final_activation(output), target, is_training)

        return output, loss, accuracy

    def _check_early_stopping(self, best_model_found):
        """
        Check current patience value and terminate if patience reached 0
        :param best_model_found: is current model the best one according to validation criterion
        :return: True if the training should be terminated, False otherwise
        """
        if best_model_found:
            self.patience = self.max_patience
        else:
            self.patience -= 1
            if self.patience <= 0 and self.num_epoch > 150:
                print('Num epochs:')
                print(self.num_epoch)
                return True
                # early stop the training
        return False

    def _is_best_val_accuracy(self, val_accuracy):
        is_best = val_accuracy > self.best_val_accuracy
        if is_best:
            self.logger.info(
                f'Saving new best validation accuracy: {val_accuracy}')
        self.best_val_accuracy = max(val_accuracy, self.best_val_accuracy)
        return is_best

    def _save_checkpoint(self, is_best):
        utils.save_checkpoint({
            'epoch': self.num_epoch,
            'num_iterations': self.num_iterations,
            'model_state_dict': self.model.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            'validate_iters': self.validate_iters,
            'max_patience': self.max_patience
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=self.logger)

    def _log_stats(self, phase, loss_avg, accuracy_avg, elapsed_time, avg_sf=None):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_accuracy_avg': accuracy_avg,
            f'{phase}_time': elapsed_time
        }
        if avg_sf is not None:
            tag_value[f'{phase}_sf'] = avg_sf

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_epoch)  # self.num_iterations)

    def _log_params(self):
        self.logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_epoch)  # self.num_iterations)
            if not (self.ternary_style and (name == 'encoders.0.double_conv.conv1.weight'
                                         or name == 'encoders.0.double_conv.binconv1.conv.weight')):
                self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(),
                                          self.num_epoch)  # self.num_iterations)

    def _log_images(self, input, target, prediction):
        values, indices = torch.max(prediction, 1)
        colored_targets = colorEncode(target[0, :, :, target.shape[3]//2].data.cpu().numpy(),
                                      self.colors, mode='RGB')
        colored_predictions = colorEncode(indices[0, :, :, indices.shape[3]//2].data.cpu().numpy(),
                                          self.colors, mode='RGB')
        colored_targets = vutils.make_grid(torch.from_numpy(colored_targets).unsqueeze(0),
                                           nrow=1, normalize=False, scale_each=True)
        colored_predictions = vutils.make_grid(torch.from_numpy(colored_predictions).unsqueeze(0),
                                               nrow=1, normalize=False, scale_each=True)
        sources = {
            'inputs': input[0, 0, :, :, input.shape[4]//2].data.cpu().numpy(),
            'targets': colored_targets.data.cpu().numpy(),
            'predictions': colored_predictions.data.cpu().numpy()
        }
        for name, batch in sources.items():
            # for tag, image in self._images_from_batch(name, batch):
            self.writer.add_image(f'{name}/last_rnd/mid_slice',
                                  batch if len(batch.shape) == 2 else batch.transpose(2, 0, 1), self.num_epoch)
            # self.num_iterations)

    def _images_from_batch(self, name, batch):
        tag_template = '{}/batch_{}/channel_{}/slice_{}'

        tagged_images = []

        if batch.ndim == 5:
            slice_idx = batch.shape[2] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                for channel_idx in range(batch.shape[1]):
                    tag = tag_template.format(name, batch_idx, channel_idx, slice_idx)
                    img = batch[batch_idx, channel_idx, slice_idx, ...]
                    tagged_images.append((tag, (self._normalize_img(img))))
        else:
            slice_idx = batch.shape[1] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                tag = tag_template.format(name, batch_idx, 0, slice_idx)
                img = batch[batch_idx, slice_idx, ...]
                tagged_images.append((tag, (self._normalize_img(img))))

        return tagged_images

    @staticmethod
    def _normalize_img(img):
        return (img - np.min(img)) / np.ptp(img)
