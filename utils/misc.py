#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:35:06 2020

@author: baekduchoi
"""

"""
    Script for miscellaneous functions used
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'.'))

import json
import torch
from torch.utils.data import DataLoader
from data import HalftoneDataset
from torch.nn import functional as F

# import cv2
import scipy.signal
import numpy as np

"""
    Function that reads the json file and generates the dataloader to be used
    Only generates training and validation dataloader
"""
def create_dataloaders(params) :
    train_img_root = params["datasets"]["train"]["root_img"]
    train_halftone_root = params["datasets"]["train"]["root_halftone"]
    batch_size = int(params["datasets"]["train"]["batch_size"])
    train_img_type = params['datasets']['train']['img_type']
    n_workers = int(params['datasets']['train']['n_workers'])
    train_use_aug = params['datasets']['train']['use_aug']
    
    val_img_root = params["datasets"]["val"]["root_img"]
    val_halftone_root = params["datasets"]["val"]["root_halftone"]
    val_img_type = params['datasets']['val']['img_type']
    
    train_dataset = HalftoneDataset(train_img_root,
                                        train_halftone_root,
                                        train_img_type,
                                        train_use_aug)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=n_workers,
                                  shuffle=True)
    
    # no need to use augmentation for validation data
    val_dataset = HalftoneDataset(val_img_root,
                                        val_halftone_root,
                                        val_img_type)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                num_workers=n_workers,
                                shuffle=False)
    
    return train_dataloader, val_dataloader

"""
    Function that reads the components of the json file and returns a dataloader for test dataset
    Refer to test_naive.json for the structure of json file
    For test dataset we do not use data augmentation

    params : output of read_json(json_file_location)
"""
def create_test_dataloaders(params) :
    test_root = params["datasets"]["test"]["dataroot"]
    batch_size = int(params["datasets"]["test"]["batch_size"])
    test_img_type = params['datasets']['test']['img_type']
    n_workers = int(params['datasets']['test']['n_workers'])
    test_img_size = int(params['datasets']['test']['img_size'])
    
    test_dataset = GrayscaleDataset(test_root,
                                    test_img_type,
                                    test_img_size,
                                    False)
    test_dataloader = DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  num_workers=n_workers,
                                  shuffle=False)
    
    return test_dataloader

"""
    Function that reads the json file
"""
def read_json(json_dir) : 
    with open(json_dir,'r') as f :
        params = json.load(f)
    return params

"""
    Nasanen's HVS model
"""
class HVS(object) :
    
    def __init__(self) :
        N = 23
        c = 0.525
        d = 3.91
        G = 11.0
        pi = np.pi
        fs = pi*3500.0/180.0
        k = fs/(c*np.log(G)+d)
        
        self.hvs = np.zeros((2*N+1,2*N+1))
        
        for i in range(2*N+1) :
            for j in range(2*N+1) :
                m = i-N
                n = j-N
                
                denom = ((k**2)+4.0*(pi**2)*((m**2)+(n**2)))**1.5                
                val = 2.0*pi*k/denom
                
                dist = (float(m)**2.0+float(n)**2.0)**0.5
                if dist > float(N) :
                    self.hvs[i][j] = 0.0
                else :
                    self.hvs[i][j] = val*(float(N)+1-dist)
                
        
        self.hvs = self.hvs/np.max(self.hvs)
        self.N = N
    
    def __getitem__(self, keys) :
        m = keys[0]+self.N
        n = keys[1]+self.N
        return self.hvs[m][n]
    
    def getHVS(self) :
        return self.hvs
    
    def size(self) :
        return self.hvs.shape

"""
    HVS error loss function
"""
def HVSloss(img1,img2,hvs) :
    k = hvs.size(2)
    M = img1.size(2)
    N = img1.size(3)

    img1_filtered = F.conv2d(img1,hvs)
    img1_filtered = img1_filtered[:,:,k-1:k+M-1,k-1:k+N-1]
    img2_filtered = F.conv2d(img2,hvs)
    img2_filtered = img2_filtered[:,:,k-1:k+M-1,k-1:k+N-1]
    

    return F.mse_loss(img1_filtered,img2_filtered)

# class Cpp(object) :
    
#     def __init__(self,hvs) :
#         self.cpp = scipy.signal.correlate2d(hvs.getHVS(),hvs.getHVS())
    
#     def __getitem__(self,keys) :
#         N = int((self.cpp.shape[0]-1)/2)
#         m = keys[0]+N
#         n = keys[1]+N
#         return self.cpp[m][n]
    
#     def getCpp(self) :
#         return self.cpp
    
#     def size(self) :
#         return self.cpp.shape

# class Cpe(object) :
#     def __init__(self,cpp,error_image) :
#         self.cpe = scipy.signal.correlate2d(error_image,cpp.getCpp()\
#                                            ,mode='same',boundary='wrap')
    
#     def __getitem__(self,keys) :
#         return self.cpe[keys[0]][keys[1]]
    
#     def updateCpe(self,m,n,val) :
#         self.cpe[m][n] = val
    
#     def getCpe(self) :
#         return self.cpe
    
#     def size(self) :
#         return self.cpe.shape

# if __name__ == '__main__' :
#     img_id = str(int(np.floor(np.random.random()*1000)))

#     hvs = HVS()
#     img_name = '../dataset_cocoval/'+img_id+'.tiff'
#     halftone_name = '../halftone_cocoval/'+img_id+'h.tiff'

#     img = cv2.imread(img_name,0).astype(np.float32)/255.0
#     imgH = cv2.imread(halftone_name,0).astype(np.float32)/255.0

#     error_img = img-imgH
#     cpp = Cpp(hvs)
    
#     cpe = Cpe(cpp,error_img)
    
#     E = np.sum(cpe.getCpe()*error_img)

#     img = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(img),0),0)
#     imgH = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(imgH),0),0)

#     hvs = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(hvs.getHVS().astype(np.float32)),0),0)
#     E2 = HVSloss(img,imgH,hvs).item()

#     print(E/256/256)
#     print(E2)

