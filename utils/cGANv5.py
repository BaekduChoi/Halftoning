import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'.'))

from torch import nn
import torch
from torch import optim
from torch.nn import functional as F
from abc import ABC
from tqdm import tqdm
import numpy as np
import cv2
import argparse

from blocks import *
from network import *
from misc import *

class cGANv5 :
    def __init__(self,json_dir,cuda=True,alpha=1.0) :

        torch.autograd.set_detect_anomaly(True)

        self.params = read_json(json_dir)
        self.device = torch.device('cuda') if cuda else torch.device('cpu')

        self.netG = StyleNetIN(2,1,ksize=5)
        # self.netG = UnetIN(1,1)
        self.netD1 = Discriminator(in_ch=1)
        self.netD2 = Discriminator2(in_ch=1)

        self.netG = self.netG.to(self.device)
        self.netD1 = self.netD1.to(self.device)
        self.netD2 = self.netD2.to(self.device)

        self.alpha = alpha

    def getparams(self) :
        # reading the hyperparameter values from the json file
        self.lr = self.params['solver']['learning_rate']
        self.lambda_hvs = self.params['solver']['lambda_hvs']
        self.lambda_GT = self.params['solver']['lambda_GT']
        self.lr_ratio1 = self.params['solver']['lr_ratio1']
        self.lr_ratio2 = self.params['solver']['lr_ratio2']
        self.beta1 = self.params['solver']['beta1']
        self.beta2 = self.params['solver']['beta2']
        self.betas = (self.beta1,self.beta2)
        self.batch_size = self.params['datasets']['train']['batch_size']
    
    def getopts(self) :
        # set up the optimizers and schedulers
        # two are needed for each - one for generator and one for discriminator
        # note for discriminator smaller learning rate is used (scaled by lr_ratio)
        self.optimizerD1 = optim.Adam(self.netD1.parameters(),lr=self.lr*self.lr_ratio1,betas=self.betas)
        self.optimizerD2 = optim.Adam(self.netD2.parameters(),lr=self.lr*self.lr_ratio2,betas=self.betas)
        self.optimizerG = optim.Adam(self.netG.parameters(),lr=self.lr,betas=self.betas)

    def train(self) :
        trainloader, valloader = create_dataloaders(self.params)

        # here we use least squares adversarial loss following cycleGAN paper
        self.gan_loss = LSGANLoss(self.device)
        # GT loss
        self.GT_loss = nn.L1Loss()

        self.inittrain()
        # num_batches is saved for normalizing the running loss
        self.num_batches = len(trainloader)

        # starting iteration
        for epoch in range(self.start_epochs,self.epochs) :
            print('Epoch = '+str(epoch+1))

            # training part of the iteration
            self.running_loss_G = 0.0
            self.running_loss_D1 = 0.0
            self.running_loss_D2 = 0.0

            # tqdm setup is borrowed from SRFBN github
            # https://github.com/Paper99/SRFBN_CVPR19
            with tqdm(total=len(trainloader),\
                    desc='Epoch: [%d/%d]'%(epoch+1,self.epochs),miniters=1) as t:
                for i,data in enumerate(trainloader) :
                    # inputG = input image in grayscale
                    inputG = data['img']
                    imgsH = data['halftone']
                    inputS = data['screened']
                    inputG = inputG.to(self.device)
                    inputS = inputS.to(self.device)
                    imgsH = imgsH.to(self.device)

                    output_vae_orig, loss_GT, loss_hvs = self.fitG(inputG,inputS,imgsH)

                    # use a pool of generated images for training discriminator
                    if self.use_pool :
                        if self.batch_size == 1 :
                            output_vae = self.pooling_onesample(output_vae_orig.detach())
                        else :
                            output_vae = self.pooling(output_vae_orig.detach())
                    else :
                        output_vae = output_vae_orig.detach()

                    loss_D1, loss_D2 = self.fitD(imgsH,output_vae,epoch)
                    
                    # tqdm update
                    t.set_postfix_str('G loss : %.4f'%(loss_GT)+\
                                    ', D1 loss : %.4f'%((loss_D1)/2)+\
                                    ', D2 loss : %.4f'%((loss_D2)/2)+\
                                    ', G HVS loss : %.4f'%(loss_hvs))
                    t.update()
                    
            # print the running L1 loss for G and adversarial loss for D when one epoch is finished        
            print('Finished training for epoch '+str(epoch+1)\
                +', D1 loss = '+str(self.running_loss_D1)+', D2 loss = '+str(self.running_loss_D2))
            
            # validation is tricky for GANs - what to use for validation?
            # since no quantitative metric came to mind, I am just saving validation results
            # visually inspecting them helps finding issues with training
            # the validation results are saved in validation path
            if valloader is not None :
                print('Validating - trying generator on random images and saving')
                self.val(valloader,self.val_path,16,epoch)

            self.saveckp(epoch)

        
    def inittrain(self) :
        self.getparams()
        self.getopts()

        # reading more hyperparameters and checkpoint saving setup
        # head_start determines how many epochs the generator will head-start learning
        self.epochs = self.params['solver']['num_epochs']
        self.head_start = self.params['solver']['head_start']
        self.save_ckp_step = self.params['solver']['save_ckp_step']
        self.pretrained_path = self.params['solver']['pretrained_path']
        self.val_path = self.params['solver']['val_path']
        self.use_pool = self.params['solver']['use_pool']

        # hvs
        hvs = HVS().getHVS().astype(np.float32)
        self.hvs = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(hvs),0),0).to(self.device)

        if self.val_path[-1] != '/':
            self.val_path += '/'

        if not os.path.isdir(self.pretrained_path) :
            os.mkdir(self.pretrained_path)
        
        if not os.path.isdir(self.val_path) :
            os.mkdir(self.val_path)

        # code for resuming training
        # if pretrain = False, the training starts from scratch as expected
        # otherwise, the checkpoint is loaded back and training is resumed
        # for the checkpoint saving format refer to the end of the function
        self.start_epochs = 0
        self.pretrain = self.params['solver']['pretrain']

        self.pool = None
        self.pool_size = 64*self.batch_size

        if self.pretrain :
            self.loadckp()    
    
    def test_final(self) :
        self.loadckp_test()

        testloader = create_test_dataloaders(self.params)
        test_path = self.params["solver"]["testpath"]
        if test_path[-1] != '/' :
            test_path += '/'

        if not os.path.isdir(test_path) :
            os.mkdir(test_path)

        self.test(testloader,test_path,save_scr=True)
    
    def loadckp_test(self) :
        self.ckp_load = torch.load(self.params['solver']['ckp_path'])
        self.netG.load_state_dict(self.ckp_load['modelG_state_dict'])

    def test(self,testloader,test_dir,save_scr=True) :
        with torch.no_grad() :
            count = 0
            with tqdm(total=len(testloader),\
                    desc='Testing.. ',miniters=1) as t:
                for ii,data in enumerate(testloader) :
                    inputG = data['img']
                    inputG = inputG.to(self.device)

                    inputS = data['screened']
                    inputS = inputS.to(self.device)

                    outputs = self.netG(torch.cat([inputG,inputS],dim=1))
                    
                    img_size1,img_size2 = outputs.shape[2], outputs.shape[3]
                    #print(outputs.shape)
                    
                    for j in range(outputs.shape[0]) :
                        imgR = torch.zeros([img_size1,img_size2],dtype=torch.float32)
                        imgR[:,:] = outputs[j,:,:].squeeze()
                        imgR = imgR.detach().numpy()
                        imgR = np.clip(imgR,0,1)
                        imgBGR = (255*imgR).astype('uint8')
                        imname = test_dir+str(count+1)+'.png'
                        cv2.imwrite(
                            imname,imgBGR)
                        imgR2 = (imgR>=0.5).astype(np.float32)
                        imgBGR2 = (255*imgR2).astype('uint8')
                        imname2 = test_dir+str(count+1)+'_thr.png'
                        cv2.imwrite(
                            imname2,imgBGR2)

                        if save_scr :
                            imgS = torch.zeros([img_size1,img_size2],dtype=torch.float32)
                            imgS[:,:] = inputS[j,0,:,:].squeeze()
                            imgS = imgS.numpy()
                            sname = test_dir+str(count+1)+'_scr.png'
                            cv2.imwrite(sname,(255*imgS).astype('uint8'))
                        
                        count += 1
                    # tqdm update
                    t.update()
    
    def val(self,testloader,test_dir,early_stop=None,epoch=None,save_scr=True) :
        with torch.no_grad() :
            count = 0
            for ii,data in enumerate(testloader) :
                inputG = data['img']
                inputG = inputG.to(self.device)

                inputS = data['screened']
                inputS = inputS.to(self.device)

                outputs = self.netG(torch.cat([inputG,inputS],dim=1))
                
                img_size1,img_size2 = outputs.shape[2], outputs.shape[3]
                #print(outputs.shape)
                
                for j in range(outputs.shape[0]) :
                    imgR = torch.zeros([img_size1,img_size2],dtype=torch.float32)
                    imgR[:,:] = outputs[j,:,:].squeeze()
                    imgR = imgR.detach().numpy()
                    imgR = np.clip(imgR,0,1)
                    imgBGR = (255*imgR).astype('uint8')
                    imname = test_dir+str(count+1)+'_epoch'+str(epoch+1)+'.png' if epoch != None else test_dir+str(count+1)+'.png'
                    cv2.imwrite(
                        imname,imgBGR)
                    imgR2 = (imgR>=0.5).astype(np.float32)
                    imgBGR2 = (255*imgR2).astype('uint8')
                    imname2 = test_dir+str(count+1)+'_epoch'+str(epoch+1)+'_thr.png' if epoch != None else test_dir+str(count+1)+'_thr.png'
                    cv2.imwrite(
                        imname2,imgBGR2)

                    if save_scr :
                        imgS = torch.zeros([img_size1,img_size2],dtype=torch.float32)
                        imgS[:,:] = inputS[j,0,:,:].squeeze()
                        imgS = imgS.numpy()
                        sname = test_dir+str(count+1)+'_scr.png'
                        cv2.imwrite(sname,(255*imgS).astype('uint8'))
                    
                    count += 1
                if early_stop != None :
                    if count >= early_stop :
                        break
    
    def saveckp(self,epoch) :
        if (epoch+1)%self.save_ckp_step == 0 :
            path = self.pretrained_path+'/ckp_epoch'+str(epoch+1)+'.ckp'
            if self.use_pool :
                torch.save({
                    'epoch':epoch+1,
                    'modelG_state_dict':self.netG.state_dict(),
                    'optimizerG_state_dict':self.optimizerG.state_dict(),
                    'modelD1_state_dict':self.netD1.state_dict(),
                    'optimizerD1_state_dict':self.optimizerD1.state_dict(),
                    'modelD2_state_dict':self.netD2.state_dict(),
                    'optimizerD2_state_dict':self.optimizerD2.state_dict(),
                    'loss_G':self.running_loss_G,
                    'loss_D1':self.running_loss_D1,
                    'loss_D2':self.running_loss_D2,
                    'pool':self.pool
                },path)
            else :
                torch.save({
                    'epoch':epoch+1,
                    'modelG_state_dict':self.netG.state_dict(),
                    'optimizerG_state_dict':self.optimizerG.state_dict(),
                    'modelD1_state_dict':self.netD1.state_dict(),
                    'optimizerD1_state_dict':self.optimizerD1.state_dict(),
                    'modelD2_state_dict':self.netD2.state_dict(),
                    'optimizerD2_state_dict':self.optimizerD2.state_dict(),
                    'loss_G':self.running_loss_G,
                    'loss_D1':self.running_loss_D1,
                    'loss_D2':self.running_loss_D2,
                },path)

    
    def loadckp(self) :
        self.ckp_load = torch.load(self.params['solver']['ckp_path'])
        self.start_epochs = self.ckp_load['epoch']
        self.netG.load_state_dict(self.ckp_load['modelG_state_dict'])
        self.netD1.load_state_dict(self.ckp_load['modelD1_state_dict'])
        self.netD2.load_state_dict(self.ckp_load['modelD2_state_dict'])
        self.optimizerG.load_state_dict(self.ckp_load['optimizerG_state_dict'])
        self.optimizerD1.load_state_dict(self.ckp_load['optimizerD1_state_dict'])
        self.optimizerD2.load_state_dict(self.ckp_load['optimizerD2_state_dict'])
        lossG_load = self.ckp_load['loss_G']
        lossD1_load = self.ckp_load['loss_D1']
        lossD2_load = self.ckp_load['loss_D2']

        if self.use_pool :
            self.pool = self.ckp_load['pool']

        print('Resumed training - epoch '+str(self.start_epochs+1)+' with G loss = '\
            +str(lossG_load)+', D1 loss = '+str(lossD1_load)+', and D2 loss = '+str(lossD2_load))

    def fitG(self,inputG,inputS,imgsH) :

        # generated image for cGAN
        output_vae_orig = self.netG(torch.cat([inputG,inputS],dim=1))

        for _p in self.netD1.parameters() :
            _p.requires_grad_(False)
            
        for _p in self.netD2.parameters() :
            _p.requires_grad_(False)

        # GAN for cGAN
        pred_fake_vae1 = self.netD1(output_vae_orig)
        pred_fake_vae2 = self.netD2(output_vae_orig)
        loss_disc_vae1 = self.gan_loss(pred_fake_vae1,True)
        loss_disc_vae2 = self.gan_loss(pred_fake_vae2,True)
        loss_disc_vae = loss_disc_vae1+loss_disc_vae2

        loss_hvs = HVSloss(output_vae_orig,inputG,self.hvs)
        loss_GT = self.GT_loss(output_vae_orig,imgsH)

        loss_G = self.lambda_GT*loss_GT+loss_disc_vae+self.lambda_hvs*loss_hvs
        
        # generator weight update
        # for the generator, all the loss terms are used
        self.optimizerG.zero_grad()

        loss_G.backward()

        torch.nn.utils.clip_grad_norm_(self.netG.parameters(),1.0)

        # backpropagation for generator and encoder
        self.optimizerG.step()
        # check only the L1 loss with GT colorization for the fitting procedure
        self.running_loss_G += loss_GT.item()/self.num_batches

        return output_vae_orig, loss_GT.item(), loss_hvs.item()

    def pooling(self,output_vae_orig) :
        if self.pool is None :
            self.pool = output_vae_orig
            output_vae = output_vae_orig
        elif self.pool.shape[0] < self.pool_size :
            self.pool = torch.cat([self.pool,output_vae_orig],dim=0)
            output_vae = output_vae_orig
        else :
            temp = output_vae_orig
            batch_size = temp.shape[0]
            ridx = torch.randperm(batch_size)
            ridx2 = torch.randperm(self.pool.shape[0])
            output_vae = torch.cat((temp[ridx[:batch_size//2],:,:,:],self.pool[ridx2[:batch_size//2],:,:,:]),dim=0)
            self.pool = torch.cat((self.pool[ridx2[batch_size//2:],:,:,:],temp[ridx[batch_size//2:],:,:,:]),dim=0)
        return output_vae
    
    def pooling_onesample(self,output_vae_orig) :
        if self.pool is None :
            self.pool = output_vae_orig
            output_vae = output_vae_orig
        else :
            if self.pool.shape[0] < self.pool_size :
                self.pool = torch.cat([self.pool,output_vae_orig],dim=0)

            prob = np.random.random_sample()
            if prob >= 0.5 :
                rindex = np.random.randint(0,self.pool.size(0))
                output_vae = torch.unsqueeze(self.pool[rindex,:,:,:],0)
                self.pool[rindex,:,:,:] = output_vae_orig
            else :
                output_vae = output_vae_orig
        return output_vae
    
    def fitD(self,imgsH,output_vae,epoch) :
        # enable the discriminator weights to be updated
        for _p in self.netD1.parameters() :
            _p.requires_grad_(True)
        for _p in self.netD2.parameters() :
            _p.requires_grad_(True)
        
        # discriminator weight update
        # for the discriminator only adversarial loss is needed
        self.optimizerD1.zero_grad()
        self.optimizerD2.zero_grad()

        pred_real1 = self.netD1(imgsH,noise=0.2)
        pred_real2 = self.netD2(imgsH,noise=0.2)
        pred_fake1 = self.netD1(output_vae,noise=0.2)
        pred_fake2 = self.netD2(output_vae,noise=0.2)

        loss_real1 = self.gan_loss(pred_real1,True)
        loss_real2 = self.gan_loss(pred_real2,True)

        loss_fake1 = self.gan_loss(pred_fake1,False)
        loss_fake2 = self.gan_loss(pred_fake2,False)

        loss_real = loss_real1+loss_real2
        loss_fake = loss_fake1+loss_fake2

        loss_D = loss_real+loss_fake

        loss_D1 = loss_real1.item()+loss_fake1.item()
        loss_D2 = loss_real2.item()+loss_fake2.item()

        # backpropagation for the discriminator
        loss_D.backward()
        torch.nn.utils.clip_grad_norm_(self.netD1.parameters(),1.0)
        torch.nn.utils.clip_grad_norm_(self.netD2.parameters(),1.0)
        # check for headstart generator learning
        if epoch >= self.head_start :
            self.optimizerD1.step()
        if epoch >= self.head_start :
            self.optimizerD2.step()

        # checking adversarial loss for the fitting procedure
        self.running_loss_D1 += (loss_D1)/2/self.num_batches
        self.running_loss_D2 += (loss_D2)/2/self.num_batches

        return loss_D1, loss_D2

    