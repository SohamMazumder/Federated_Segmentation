#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def volume_iid(volumes, num_users):
    """
    Sample I.I.D. client volumes
    :param volume:
    :param num_users:
    :return: dict of volume index
    """
    num_items = int(len(volumes)/num_users)
    dict_users, all_volumes = {}, [i for i in range(len(volumes))]
    # print(len(dataset))
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_volumes, num_items, replace=False))
        all_volumes = list(set(all_volumes) - dict_users[i])
    # print(dict_users)
    return dict_users


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    print(len(dataset))
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    #print(dict_users)
    return dict_users

def non_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    x=0
    y=num_items
    for i in range(num_users):
        dict_users[i] = set(all_idxs[x:y])
        x += num_items
        y += num_items
    return dict_users

if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
