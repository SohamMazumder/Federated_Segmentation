    #!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics

from nn_common_modules import losses as additional_losses

from utils.evaluator import evaluate_dice_score
from models.test import log_images
#from utils.3D_Losses import
torch.set_default_tensor_type('torch.FloatTensor')


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label, w = self.dataset[self.idxs[item]]
        return image, label, w
    

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, logwriter=None, user_id=None, testLoader = None, epoch = None):
        self.args = args
        self.logwriter = logwriter
        self.user_id = user_id
        self.test_loader = testLoader
        self.epoch = epoch
        #weights = [1.80434783, 0.34583333, 1.25757576, 0.84693878, 1.06410256, 1.09210526, 0.25, 3.45833333, 0.32170543, 0.94318182]
        #class_weights = torch.FloatTensor(weights).cuda()
        self.loss_func = additional_losses.CombinedLoss()
        self.selected_clients = []
        #self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True)
    def train(self, net):
        net.train()
        # train and update
        #optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            print("Local Epoch: " + str(iter+1))
            batch_loss = []
            for batch_idx, (images, labels, w) in enumerate(self.ldr_train):
                #print(images.shape)
                #print(labels.shape)
                #print(w.shape)
                #print(w)
                images, labels, w = images.type(torch.FloatTensor), labels.type(torch.LongTensor), w.type(torch.FloatTensor)
                images, labels, w = images.to(self.args.device), labels.to(self.args.device), w.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                #print(log_probs)
                #print(log_probs.shape)
                loss = self.loss_func(log_probs, labels, w)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            local_loss = sum(batch_loss)/len(batch_loss)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            self.logwriter.add_scalar('Train_Loss_User{:3d}'.format(self.user_id),  local_loss, (iter+ self.args.local_ep*self.epoch))
            
            #Evaluation
            dice_score,_ = evaluate_dice_score(net, '/data/OASISchallenge/FS/', '/data/OASISchallenge/',
                        '/data/OASISchallenge/testing_15.txt', self.args.log_folder)
            self.test_img(net, self.test_loader, self.logwriter, self.args, self.user_id, (iter+ self.args.local_ep*self.epoch))
            
            self.logwriter.add_scalar('Dice_Score_User{:3d}'.format(self.user_id),  dice_score, (iter+ self.args.local_ep*self.epoch))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def test_img(self, net_g, dataLoader, logwriter,args,user, step):
        net_g.eval()
        # testing
        test_loss = 0
        data_loader = dataLoader
        loss_func = additional_losses.CombinedLoss()
        for batch_idx, (images, labels, w) in enumerate(data_loader):
            images, labels, w = images.type(torch.FloatTensor), labels.type(torch.LongTensor), w.type(torch.FloatTensor)
            if args.gpu != -1:
                images, labels, w = images.cuda(), labels.cuda(), w.cuda()
            log_probs = net_g(images)
            #if (batch_idx%500 ==0 and (step - self.epoch)%3 == 0):
                #log_images(images, labels, log_probs,logwriter, step)
            # sum up batch loss
            loss = loss_func(log_probs, labels, w).item()
            test_loss += loss
        test_loss /= len(data_loader.dataset)
        logwriter.add_scalar('Test_Loss_User {:3d}'.format(user), test_loss, step)


