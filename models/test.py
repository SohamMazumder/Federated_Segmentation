#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from nn_common_modules import losses as additional_losses
import matplotlib.pyplot as plt
import numpy as np
import os

def test_img(net_g, datatest, logwriter,args, step):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = datatest
    loss_func = additional_losses.CombinedLoss()
    for batch_idx, (images, labels, w) in enumerate(data_loader):
        images, labels, w = images.type(torch.FloatTensor), labels.type(torch.LongTensor), w.type(torch.FloatTensor)
        if args.gpu != -1:
            images, labels, w = images.cuda(), labels.cuda(), w.cuda()
        log_probs = net_g(images)
        
        if (batch_idx%200 ==0):
            log_images(images, labels, log_probs,logwriter,step, batch_idx)
        # sum up batch loss
        loss = loss_func(log_probs, labels, w).item()
        test_loss += loss
        
        #logwriter.add_scalar('Test Loss', loss, batch_idx)

        # get the index of the max log-probability
        
    test_loss /= len(data_loader.dataset)
    logwriter.add_scalar('Test Loss', test_loss, step)
    #accuracy = 100.00 * correct / len(data_loader.dataset)
    #if args.verbose:
        #print('\nTest set: Average loss: {:.4f} \n'.format(test_loss))


def dice_score_perclass(vol_output, ground_truth, num_classes=33, no_samples=10, mode='train'):
    dice_perclass = torch.zeros(num_classes)
    if mode == 'train':
        samples = np.random.choice(len(vol_output), no_samples)
        vol_output, ground_truth = vol_output[samples], ground_truth[samples]
    for i in range(num_classes):
        GT = (ground_truth == i).float()
        Pred = (vol_output == i).float()
        inter = torch.sum(torch.mul(GT, Pred))
        union = torch.sum(GT) + torch.sum(Pred) + 0.0001
        dice_perclass[i] = (2 * torch.div(inter, union))
    return dice_perclass

from scipy.io import loadmat
from torchvision import utils as vutils
color_path = os.getcwd() + "/utils/color150.mat"
colors = loadmat(color_path)['colors']
colors[0] = [0, 0, 0]

def log_images(input=None, target=None, prediction=None,logwriter=None,iter=1, batch_num = None):
    values, indices = torch.max(prediction, 1)
    colored_targets = colorEncode(target[0, :, :].data.cpu().numpy(),
                                  colors, mode='RGB')
    colored_predictions = colorEncode(indices[0, :, :].data.cpu().numpy(),
                                      colors, mode='RGB')
    colored_targets = vutils.make_grid(torch.from_numpy(colored_targets).unsqueeze(0),
                                       nrow=1, normalize=False, scale_each=True)
    colored_predictions = vutils.make_grid(torch.from_numpy(colored_predictions).unsqueeze(0),
                                           nrow=1, normalize=False, scale_each=True)
    sources = {
        'inputs': input[0, 0, :, :].data.cpu().numpy(),
        'targets': colored_targets.data.cpu().numpy(),
        'predictions': colored_predictions.data.cpu().numpy()
    }
    for name, batch in sources.items():
        # for tag, image in self._images_from_batch(name, batch):
        logwriter.add_image(f'{name}:{batch_num}',
                              batch if len(batch.shape) == 2 else batch.transpose(2, 0, 1),iter)
        # self.num_iterations)

def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors, mode='BGR'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb
