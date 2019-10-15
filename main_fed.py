#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils.sampling import volume_iid, mnist_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import QuickNat
from models.Fed import FedAvg
from models.test import test_img, log_images


from utils.loadData import load_data_h5
from utils.evaluator import evaluate_dice_score
import utils.data_utils as du

torch.set_default_tensor_type('torch.FloatTensor')

if __name__ == '__main__':
    
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # tensorboard init
    writer = SummaryWriter(args.log_folder)

    if args.loss_balance == 1:
        print("Loss balancing activated")
    
    # load dataset and split users
    if args.dataset == 'brain':
        # load and distribute malc dataset
        if args.malc:
            data_params = {'data_dir':'/data/OASISchallenge/FS/',
                           'label_dir':'/data/OASISchallenge/',
                           'train_volumes': '/data/OASISchallenge/training_15.txt',
                           'test_volumes':'/data/OASISchallenge/testing_15.txt'}

            train_file_paths = du.load_file_paths(data_params['data_dir'], data_params['label_dir'], data_params['train_volumes'])
            test_file_paths = du.load_file_paths(data_params['data_dir'], data_params['label_dir'], data_params['test_volumes'])
            for i in range(5):
                train_file_paths.append(test_file_paths[i])
            test_file_paths = test_file_paths[5:]
            print(len(train_file_paths))
            print(len(test_file_paths))
            # train_set, test_set = load_data_h5('/data/OASISchallenge/FS/', '/data/OASISchallenge/','/data/OASISchallenge/training_15.txt',
                                           #'/data/OASISchallenge/testing_15.txt')

            # sample users
            dict_users = volume_iid(train_file_paths, args.num_users)

        # add ibsr dataset
        if args.ibsr:
            print("Loading IBSR")
            # IBSR_08 volume creates a problem so easier to remove it
            # ibsr_paths = du.load_ibsr_paths(data_dir='/home/soham/ibsr/', volumes_txt_file='/home/soham/ibsr/training.txt')
            ibsr_paths = du.load_ibsr_paths(data_dir='/data/ibsr/', volumes_txt_file='/data/ibsr/training.txt')
            num_ibsr_users = 2

            # sample users
            dict_ibsr_users = volume_iid(ibsr_paths, num_ibsr_users)

    else:
        exit('Error: unrecognized dataset')

    # build model
    if args.model == 'segment' and args.dataset == 'brain':
        params = {'num_channels':1,
                        'num_filters':64,
                        'kernel_h':5,
                        'kernel_w':5,
                        'kernel_c':1,
                        'stride_conv':1,
                        'pool':2,
                        'stride_pool':2,
                        'num_class':33,
                        'se_block': "CSSE",
                        'drop_out':0.0}

        net_glob = QuickNat(params = params).to(args.device)
    elif args.model == 'resnet' and args.dataset == 'dermo':
        model = ResnetDermo(args=args)
        net_glob = model.run()
        net_glob = net_glob.to(args.device)
    else:
        exit('Error: unrecognized model')

    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    val_acc_list, net_list = [], []
    test_set = []
    dice_score_best = 0
    start_epoch = 0
    
    # Load Testset
    # Oasis
    test_set = load_data_h5(test_file_paths = test_file_paths)
    test_loader = DataLoader(test_set, batch_size=1)

    model_path = '{}/checkpoint_best.pth'.format(args.log_folder)
    
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        best_dice = checkpoint['best_dice']
        net_glob.load_state_dict(checkpoint['model_state_dict'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(model_path))
    
    if args.mode == 'IIL':
        args.epochs = 1
    
    for iter in range(start_epoch, args.epochs):
        print("Global Epoch:", iter +1)
        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            print("User:{:3d}".format(idx))
            user_train_paths, user_train_set= [], []
            for i in dict_users[idx]:
                user_train_paths.append(train_file_paths[i])
                
            user_train_set = load_data_h5(train_file_paths = user_train_paths)
            local = LocalUpdate(args=args, dataset=user_train_set, idxs=dict_users[idx], logwriter=writer, user_id=idx, testLoader = test_loader, epoch = iter)
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        old_loc_epoch = args.local_ep
        if args.ibsr:
            args.local_ep = 1
            m = max(int(args.frac * num_ibsr_users), 1)
            ibsr_users = np.random.choice(range(num_ibsr_users), m, replace=False)
            for idx in ibsr_users:
                print("IBSR User:{:3d}".format(idx))
                user_train_paths, user_train_set= [], []
                for i in dict_ibsr_users[idx]:
                    user_train_paths.append(ibsr_paths[i])

                user_train_set = load_data_h5(train_file_paths = user_train_paths, remap_config='FS', orientation='ibsr')
                local = LocalUpdate(args=args, dataset=user_train_set, idxs=dict_ibsr_users[idx], logwriter=writer, user_id=(args.num_users+idx), testLoader=test_loader, epoch=iter)
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
            args.local_ep = old_loc_epoch

        # Median balancing based on loss
        if args.loss_balance == 1:
            losses = np.array(loss_locals)
            med_loss = np.median(losses)
            weights = np.ones_like(losses)
            for i, l in enumerate(losses):
                weights[i] = weights[i] * (med_loss/l)
                for k, v in w_locals[i].items():
                    w_locals[i][k] = v * weights[i]

                writer.add_scalar('Weight_User{:3d}'.format(i), weights[i], iter)

        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        # change num of testing accordingly in below function
        dice_score_glob,_ = evaluate_dice_score(net_glob, '/data/OASISchallenge/FS/', '/data/OASISchallenge/',
                        '/data/OASISchallenge/testing_15.txt', args.log_folder)
        
        writer.add_scalar('Dice_Score_Global',  dice_score_glob, iter)
        writer.add_scalar('Train Loss', loss_avg, iter)
        test_img(net_glob, test_loader, writer, args, iter)

        if dice_score_best< dice_score_glob:
            dice_score_best = dice_score_glob
            torch.save({
                'epoch': iter,
                'model_state_dict': net_glob.state_dict(),
                'loss': loss_avg,
                'best_dice': dice_score_best}, model_path)

    # testing
    print("Evaluation")

    net_glob.eval()

    dice_score_glob,_ = evaluate_dice_score(net_glob, '/data/OASISchallenge/FS/', '/data/OASISchallenge/',
                        '/data/OASISchallenge/testing_15.txt', args.log_folder)
    
    writer.close()
