# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:49:41 2020

@author: baekd
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'.'))

import torch
from torch import optim
from utils.network import Discriminator, Discriminator2, LSGANLoss, StyleNetIN
from utils.misc import read_json, create_dataloaders, HVS, HVSloss
from tqdm import tqdm
import numpy as np
import cv2
import argparse


"""
    Training function for cGAN based colorization
    Instead of regressing ab component, we directly generate RGB images here
    The result could be improved if we generate ab component of the image (to be implemented)
    Loss function is composed of three terms
    1) Adversarial loss : following cycleGAN, here we use least-squares adversarial loss
       Results could improve if we use Wasserstein GAN + gradient penalty (to be implemented)
    2) grayscale-consistency loss : the generated RGB image is converted back to grayscale,
       and then L1 loss is used between the grayscale converted image and input image
       To not use grayscale-consistency loss simply set lambda2 in json file as 0
    3) L1 loss between GT and generated image
    For each L1 term in the loss, scaling parameter can be set in the json file

    netG : generator network
    netD : discriminator network. assuming patchGAN-like structure from pix2pix or cycleGAN
    netE : encoder network
    device : for CUDA support
    train_dataloader : dataloader for training data
    val_dataloader : dataloader for validation data
    params : output of utils.misc.read_json(json_file_location)
"""
def train_cGAN(netG,netD1,netD2,device,train_dataloader,val_dataloader,params) :
    # generator and discriminators using CUDA
    netG = netG.to(device)
    netD1 = netD1.to(device)
    netD2 = netD2.to(device)

    # here we use least squares adversarial loss following cycleGAN paper
    gan_loss = LSGANLoss(device)
    # GT loss
    GT_loss = torch.nn.L1Loss()

    # hvs
    hvs = HVS().getHVS().astype(np.float32)
    hvs = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(hvs),0),0).to(device)
    
    # reading the hyperparameter values from the json file
    lr = params['solver']['learning_rate']
    lr_steps = params['solver']['lr_steps']
    lr_gamma = params['solver']['lr_gamma']
    lambda_GT = params['solver']['lambda_GT']
    lambda_hvs = params['solver']['lambda_hvs']
    lr_ratio1 = params['solver']['lr_ratio1']
    lr_ratio2 = params['solver']['lr_ratio2']
    beta1 = params['solver']['beta1']
    beta2 = params['solver']['beta2']
    betas = (beta1,beta2)
    
    # set up the optimizers and schedulers
    # two are needed for each - one for generator and one for discriminator
    # note for discriminator smaller learning rate is used (scaled by lr_ratio)
    optimizerD1 = optim.Adam(netD1.parameters(),lr=lr*lr_ratio1,betas=betas)
    optimizerD2 = optim.Adam(netD2.parameters(),lr=lr*lr_ratio2,betas=betas)
    optimizerG = optim.Adam(netG.parameters(),lr=lr,betas=betas)
    schedulerD1 = optim.lr_scheduler.MultiStepLR(optimizerD1,lr_steps,lr_gamma)
    schedulerD2 = optim.lr_scheduler.MultiStepLR(optimizerD2,lr_steps,lr_gamma)
    schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG,lr_steps,lr_gamma)
    
    # reading more hyperparameters and checkpoint saving setup
    # head_start determines how many epochs the generator will head-start learning
    epochs = params['solver']['num_epochs']
    head_start = params['solver']['head_start']
    save_ckp_step = params['solver']['save_ckp_step']
    pretrained_path = params['solver']['pretrained_path']
    val_path = params['solver']['val_path']
    use_pool = params['solver']['use_pool']

    if val_path[-1] != '/':
        val_path += '/'

    if not os.path.isdir(pretrained_path) :
        os.mkdir(pretrained_path)
    
    if not os.path.isdir(val_path) :
        os.mkdir(val_path)

    # code for resuming training
    # if pretrain = False, the training starts from scratch as expected
    # otherwise, the checkpoint is loaded back and training is resumed
    # for the checkpoint saving format refer to the end of the function
    start_epochs = 0
    pretrain = params['solver']['pretrain']

    pool = None
    pool_size = 64*params['datasets']['train']['batch_size']
    
    if pretrain :
        ckp_load = torch.load(params['solver']['ckp_path'])
        start_epochs = ckp_load['epoch']
        netG.load_state_dict(ckp_load['modelG_state_dict'])
        netD.load_state_dict(ckp_load['modelD_state_dict'])
        optimizerG.load_state_dict(ckp_load['optimizerG_state_dict'])
        optimizerD1.load_state_dict(ckp_load['optimizerD1_state_dict'])
        optimizerD2.load_state_dict(ckp_load['optimizerD2_state_dict'])
        schedulerG.load_state_dict(ckp_load['schedulerG_state_dict'])
        schedulerD1.load_state_dict(ckp_load['schedulerD1_state_dict'])
        schedulerD2.load_state_dict(ckp_load['schedulerD2_state_dict'])
        lossG_load = ckp_load['loss_G']
        lossD_load = ckp_load['loss_D']

        if use_pool :
            pool = ckp_load['pool']

        print('Resumed training - epoch '+str(start_epochs+1)+' with G loss = '\
            +str(lossG_load)+' and D loss = '+str(lossD_load))

    # num_batches is saved for normalizing the running loss
    num_batches = len(train_dataloader)

    # starting iteration
    for epoch in range(start_epochs,epochs) :
        print('Epoch = '+str(epoch+1))

        # training part of the iteration
        running_loss_G = 0.0
        running_loss_D = 0.0

        # tqdm setup is borrowed from SRFBN github
        # https://github.com/Paper99/SRFBN_CVPR19
        with tqdm(total=len(train_dataloader),\
                  desc='Epoch: [%d/%d]'%(epoch+1,epochs),miniters=1) as t:
            for i,data in enumerate(train_dataloader) :
                # inputG = input image in grayscale
                # imgsRGB = GT colorized image
                inputG = data['img']
                imgsH_orig = data['halftone']
                imgsS_orig = data['screened']
                inputG = inputG.to(device)
                imgsH_orig = imgsH_orig.to(device)
                imgsS_orig = imgsS_orig.to(device)

                # label smoothing if needed
                alpha = 0.0
                imgsH = (1-alpha)*imgsH_orig+alpha*torch.ones_like(imgsH_orig)
                imgsS = (1-alpha)*imgsS_orig+alpha*torch.ones_like(imgsS_orig)

                # concatenation of input image and screened image
                inputGcat = torch.cat([inputG,imgsS],1)

                # generated image for cGAN
                output_orig = netG(inputGcat)

                # encoder loss - cGAN
                loss_GT = GT_loss(output_orig,imgsH)
                loss_hvs = HVSloss(output_orig,inputG,hvs)

                for _p in netD1.parameters() :
                    _p.requires_grad_(False)

                for _p in netD2.parameters() :
                    _p.requires_grad_(False)

                # GAN for cGAN
                pred_fake1 = netD1(output_orig)
                loss_disc1 = gan_loss(pred_fake1,True,noise=0.0)

                # GAN for cGAN
                pred_fake2 = netD2(output_orig)
                loss_disc2 = gan_loss(pred_fake2,True,noise=0.0)

                loss_G = lambda_GT*loss_GT+loss_disc1+loss_disc2+lambda_hvs*loss_hvs
                
                # generator weight update
                # for the generator, all the loss terms are used
                optimizerG.zero_grad()
                
                loss_G.backward()

                torch.nn.utils.clip_grad_norm_(netG.parameters(),1.0)

                # backpropagation for generator and encoder
                optimizerG.step()
                # check only the L1 loss with GT colorization for the fitting procedure
                running_loss_G += loss_GT.item()/num_batches

                # use a pool of generated images for training discriminator
                if use_pool :
                    if pool is None :
                        pool = output_orig.detach()
                        output = output_orig.detach()
                    elif pool.shape[0] < pool_size :
                        pool = torch.cat([pool,output_orig.detach()],dim=0)
                        output = output_orig.detach()
                    else :
                        temp = output_orig.detach()
                        batch_size = temp.shape[0]
                        ridx = torch.randperm(batch_size)
                        output = torch.cat((temp[ridx[:batch_size//2],:,:,:],\
                            pool[ridx[:batch_size//2],:,:,:]),dim=0)
                        pool = torch.cat((pool[ridx[:batch_size//2],:,:,:],\
                            temp[ridx[:batch_size//2],:,:,:]),dim=0)
                else :
                    output = output_orig.detach()

                # enable the discriminator weights to be updated
                for _p in netD1.parameters() :
                    _p.requires_grad_(True)

                # enable the discriminator weights to be updated
                for _p in netD2.parameters() :
                    _p.requires_grad_(True)
                
                # discriminator weight update
                # for the discriminator only adversarial loss is needed
                optimizerD1.zero_grad()
                pred_real1 = netD1(imgsH)
                pred_fake1 = netD1(output)
                loss_D1 = gan_loss(pred_real1,True,noise=0.2)+\
                    gan_loss(pred_fake1,False,noise=0.2)

                # backpropagation for the discriminator
                loss_D1.backward()
                # check for headstart generator learning
                if epoch >= head_start :
                    optimizerD1.step()
                
                # discriminator weight update
                # for the discriminator only adversarial loss is needed
                optimizerD2.zero_grad()
                pred_real2 = netD2(imgsH)
                pred_fake2 = netD2(output)
                loss_D2 = gan_loss(pred_real2,True,noise=0.2)+\
                    gan_loss(pred_fake2,False,noise=0.2)

                # backpropagation for the discriminator
                loss_D2.backward()
                # check for headstart generator learning
                if epoch >= head_start :
                    optimizerD2.step()

                # checking adversarial loss for the fitting procedure
                running_loss_D += (loss_D1.item()+loss_D2.item())/4/num_batches
                
                # tqdm update
                t.set_postfix_str('G loss : %.2f'%(loss_GT.item())+\
                                ', D1 loss : %.2f'%(loss_D1.item()/2)+\
                                ', D2 loss : %.2f'%(loss_D2.item()/2))
                t.update()
                
        # print the running L1 loss for G and adversarial loss for D when one epoch is finished        
        print('Finished training for epoch '+str(epoch+1)\
              +', average D loss = '+str(running_loss_D))
        
        # validation is tricky for GANs - what to use for validation?
        # since no quantitative metric came to mind, I am just saving validation results
        # visually inspecting them helps finding issues with training
        # the validation results are saved in validation path
        print('Validating - trying generator on random images and saving')
        with torch.no_grad() :
            for __,data in enumerate(val_dataloader) :
                inputG = data['img']
                inputG = inputG.to(device)
                inputS = data['screened']
                inputS = inputS.to(device)

                # concatenation
                inputG2 = torch.cat([inputG,inputS],1)

                outputs = netG(inputG2)
                
                img_size = outputs.shape[2]
                
                for j in range(outputs.shape[0]) :
                    imgR = torch.zeros([img_size,img_size],dtype=torch.float32)
                    imgR[:,:] = outputs[j,0,:,:].squeeze()
                    imgR = imgR.detach().numpy()
                    imgR = np.clip(imgR,0,1)
                    imgR = (255*imgR).astype('uint8')
                    cv2.imwrite(
                        val_path+str(j+1)+'_epoch'+str(epoch+1)+'.png',imgR)
                break

        # scheduler updates the learning rate
        schedulerG.step()
        schedulerD1.step()
        schedulerD2.step()

        # saving checkpoint when necessary
        if (epoch+1)%save_ckp_step == 0 :
            path = pretrained_path+'/ckp_epoch'+str(epoch+1)+'.ckp'
            if use_pool :
                torch.save({
                    'epoch':epoch+1,
                    'modelG_state_dict':netG.state_dict(),
                    'optimizerG_state_dict':optimizerG.state_dict(),
                    'schedulerG_state_dict':schedulerG.state_dict(),
                    'modelD1_state_dict':netD1.state_dict(),
                    'optimizerD1_state_dict':optimizerD1.state_dict(),
                    'schedulerD1_state_dict':schedulerD1.state_dict(),
                    'modelD2_state_dict':netD2.state_dict(),
                    'optimizerD2_state_dict':optimizerD2.state_dict(),
                    'schedulerD2_state_dict':schedulerD2.state_dict(),
                    'loss_G':running_loss_G,
                    'loss_D':running_loss_D,
                    'pool':pool
                },path)
            else :
                torch.save({
                    'epoch':epoch+1,
                    'modelG_state_dict':netG.state_dict(),
                    'optimizerG_state_dict':optimizerG.state_dict(),
                    'schedulerG_state_dict':schedulerG.state_dict(),
                    'modelD1_state_dict':netD1.state_dict(),
                    'optimizerD1_state_dict':optimizerD1.state_dict(),
                    'schedulerD1_state_dict':schedulerD1.state_dict(),
                    'modelD2_state_dict':netD2.state_dict(),
                    'optimizerD2_state_dict':optimizerD2.state_dict(),
                    'schedulerD2_state_dict':schedulerD2.state_dict(),
                    'loss_G':running_loss_G,
                    'loss_D':running_loss_D,
                },path)

"""
    main entry for the script
"""
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt',type=str,required=True)
    args = parser.parse_args()
    json_dir = args.opt

    torch.autograd.set_detect_anomaly(True)
    params = read_json(json_dir)
    trainloader, valloader = create_dataloaders(params)
    netG = StyleNetIN(in_ch=2,out_nch=1,dn_lv=1)
    netD1 = Discriminator(in_ch=1)
    netD2 = Discriminator2()
    device = torch.device('cuda')
    train_cGAN(netG,netD1,netD2,device,trainloader,valloader,params)
            
            
    
    


    







































