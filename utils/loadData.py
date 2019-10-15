"""
Convert to h5 utility.
Sample command to create new dataset
- python utils/convert_h5.py -dd /home/masterthesis/shayan/nas_drive/Data_Neuro/OASISchallenge/FS -ld /home/masterthesis/shayan/nas_drive/Data_Neuro/OASISchallenge -trv datasets/train_volumes.txt -tev datasets/test_volumes.txt -rc Neo -o COR -df datasets/MALC/coronal
- python utils/convert_h5.py -dd /home/masterthesis/shayan/nas_drive/Data_Neuro/IXI/IXI_FS -ld /home/masterthesis/shayan/nas_drive/Data_Neuro/IXI/IXI_FS -ds 98,2 -rc FS -o COR -df datasets/IXI/coronal
"""

import argparse
import os

import h5py
import numpy as np
from torchvision import transforms
import torch

#import common_utils
import utils.data_utils as du
from utils.data_utils import ImdbData
import utils.preprocessor as preprocessor

transform_train = transforms.Compose([
    transforms.RandomCrop(200, padding=56),
    transforms.ToTensor(),
])


def load_data_h5(train_file_paths=None, test_file_paths=None, remap_config='Neo', orientation=preprocessor.ORIENTATION['coronal']):
    # Data splitting
    print("START")
    #if train_volumes and test_volumes:
        #train_file_paths = du.load_file_paths(data_dir, label_dir, train_volumes)
        #test_file_paths = du.load_file_paths(data_dir, label_dir, test_volumes)
    #else:
        #raise ValueError('You must provide a train, train dataset list')

    if train_file_paths:
        print("Train dataset size: %d" % (len(train_file_paths)))
        # loading,pre-processing and writing train data
        print("===Train data===")
        data_train, label_train, class_weights_train, weights_train, _ = du.load_dataset(train_file_paths,
                                                                                        orientation,
                                                                                        remap_config=remap_config,
                                                                                        return_weights=True,
                                                                                        reduce_slices=True,
                                                                                        remove_black=True)
        no_slices, H, W = data_train[0].shape
        data_train=np.concatenate(data_train).reshape((-1, H, W))
        label_train=np.concatenate(label_train).reshape((-1, H, W))
        class_weights_train=np.concatenate(class_weights_train).reshape((-1, H, W))
        
        print("END")  
  
        return (ImdbData(data_train, label_train, class_weights_train, transforms=transform_train))

    if test_file_paths:
        #_write_h5(data_train, label_train, class_weights_train, weights_train, f, mode='train')
        print("Test dataset size: %d" % (len(test_file_paths)))
        # loading,pre-processing and writing test data
        print("===Test data===")
        data_test, label_test, class_weights_test, weights_test, _ = du.load_dataset(test_file_paths,
                                                                                    orientation,
                                                                                    remap_config=remap_config,
                                                                                    return_weights=True,
                                                                                    reduce_slices=True,
                                                                                    remove_black=True)
        
        no_slices, H, W = data_test[0].shape
        data_test=np.concatenate(data_test).reshape((-1, H, W))
        label_test=np.concatenate(label_test).reshape((-1, H, W))
        class_weights_test=np.concatenate(class_weights_test).reshape((-1, H, W))
        
        print("END")  
  
        return (ImdbData(data_test, label_test, class_weights_test))
    else:
        raise ValueError('You must provide a train or test dataset list')


    
    #_write_h5(data_test, label_test, class_weights_test, weights_test, f, mode='test')
    
   
    
  
