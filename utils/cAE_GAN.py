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

class cAE_GAN :
    def __init__(self,json_dir,cuda=True,alpha=1.0,middle=True) :

        torch.autograd.set_detect_anomaly(True)

        self.params = read_json(json_dir)
        self.latent_dim = self.params['solver']['latent_dim']
        self.device = torch.device('cuda') if cuda else torch.device('cpu')
        self.middle = middle

        self.netG = StyleNetIN(1+self.latent_dim,1)
        self.netD = PyramidDisc(middle=self.middle)
        # self.netE = EncoderWB(in_ch=1,out_nc=self.latent_dim,vae=False,norm=None)
        self.netE = EncoderWB2(in_ch=2,out_nc=self.latent_dim,vae=False,norm=None)

        self.netG = self.netG.to(self.device)
        self.netD = self.netD.to(self.device)
        self.netE = self.netE.to(self.device)

        self.alpha = alpha

    def getparams(self) :
        # reading the hyperparameter values from the json file
        self.lr = self.params['solver']['learning_rate']
        self.lr_steps = self.params['solver']['lr_steps']
        self.lr_gamma = self.params['solver']['lr_gamma']
        self.lambda_hvs = self.params['solver']['lambda_hvs']
        self.lr_ratio = self.params['solver']['lr_ratio']
        self.lambda_latent = self.params['solver']['lambda_latent']
        self.beta1 = self.params['solver']['beta1']
        self.beta2 = self.params['solver']['beta2']
        self.betas = (self.beta1,self.beta2)
        self.batch_size = self.params['datasets']['train']['batch_size']
    
    def getopts(self) :
        # set up the optimizers and schedulers
        # two are needed for each - one for generator and one for discriminator
        # note for discriminator smaller learning rate is used (scaled by lr_ratio)
        self.optimizerD = optim.Adam(self.netD.parameters(),lr=self.lr*self.lr_ratio,betas=self.betas)
        self.optimizerG = optim.Adam(self.netG.parameters(),lr=self.lr,betas=self.betas)
        self.optimizerE = optim.Adam(self.netE.parameters(),lr=self.lr,betas=self.betas)
        self.schedulerD = optim.lr_scheduler.MultiStepLR(self.optimizerD,self.lr_steps,self.lr_gamma)
        self.schedulerG = optim.lr_scheduler.MultiStepLR(self.optimizerG,self.lr_steps,self.lr_gamma)
        self.schedulerE = optim.lr_scheduler.MultiStepLR(self.optimizerE,self.lr_steps,self.lr_gamma)

    def train(self) :
        trainloader, valloader = create_dataloaders(self.params)

        # here we use least squares adversarial loss following cycleGAN paper
        self.gan_loss = LSGANLoss(self.device)
        # cycle_loss is probably not the best name for this..
        self.cycle_loss = nn.L1Loss()
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
            self.running_loss_D = 0.0

            # tqdm setup is borrowed from SRFBN github
            # https://github.com/Paper99/SRFBN_CVPR19
            with tqdm(total=len(trainloader),\
                    desc='Epoch: [%d/%d]'%(epoch+1,self.epochs),miniters=1) as t:
                for i,data in enumerate(trainloader) :
                    # inputG = input image in grayscale
                    inputG = data['img']
                    imgsH = data['halftone']
                    inputG = inputG.to(self.device)
                    imgsH = imgsH.to(self.device)

                    output_vae_orig, loss_GT, loss_z = self.fitGE(inputG)

                    # use a pool of generated images for training discriminator
                    if self.use_pool :
                        if self.batch_size == 1 :
                            output_vae = self.pooling_onesample(output_vae_orig.detach())
                        else :
                            output_vae = self.pooling(output_vae_orig.detach())
                    else :
                        output_vae = output_vae_orig.detach()

                    loss_D = self.fitD(imgsH,output_vae,epoch)
                    
                    # tqdm update
                    t.set_postfix_str('B G HVS loss : %.4f'%(loss_GT)+\
                                    ', B D loss : %.4f'%((loss_D)/2)+\
                                    ', z loss : %.4f'%(loss_z))
                    t.update()
                    
            # print the running L1 loss for G and adversarial loss for D when one epoch is finished        
            print('Finished training for epoch '+str(epoch+1)\
                +', D loss = '+str(self.running_loss_D))
            
            # validation is tricky for GANs - what to use for validation?
            # since no quantitative metric came to mind, I am just saving validation results
            # visually inspecting them helps finding issues with training
            # the validation results are saved in validation path
            if valloader is not None :
                print('Validating - trying generator on random images and saving')
                self.test(valloader,self.val_path,16,epoch)

            # scheduler updates the learning rate
            self.schedulerG.step()
            self.schedulerE.step()
            self.schedulerD.step()

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
    
    def test(self,testloader,test_dir,early_stop=None,epoch=None) :
        with torch.no_grad() :
            count = 0
            for ii,data in enumerate(testloader) :
                inputG = data['img']
                inputG = inputG.to(self.device)

                # first sample z from N(0,I)
                z = torch.randn(inputG.size(0),self.netE.nc).to(self.device)
                z_img = z.view(z.size(0),z.size(1),1,1).\
                    expand(z.size(0),z.size(1),inputG.size(2),inputG.size(3))

                outputs = self.netG(torch.cat([inputG,z_img],1))
                
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
                    'schedulerG_state_dict':self.schedulerG.state_dict(),
                    'modelD_state_dict':self.netD.state_dict(),
                    'optimizerD_state_dict':self.optimizerD.state_dict(),
                    'schedulerD_state_dict':self.schedulerD.state_dict(),
                    'modelE_state_dict':self.netE.state_dict(),
                    'optimizerE_state_dict':self.optimizerE.state_dict(),
                    'schedulerE_state_dict':self.schedulerE.state_dict(),
                    'loss_G':self.running_loss_G,
                    'loss_D':self.running_loss_D,
                    'pool':self.pool
                },path)
            else :
                torch.save({
                    'epoch':epoch+1,
                    'modelG_state_dict':self.netG.state_dict(),
                    'optimizerG_state_dict':self.optimizerG.state_dict(),
                    'schedulerG_state_dict':self.schedulerG.state_dict(),
                    'modelD_state_dict':self.netD.state_dict(),
                    'optimizerD_state_dict':self.optimizerD.state_dict(),
                    'schedulerD_state_dict':self.schedulerD.state_dict(),
                    'modelE_state_dict':self.netE.state_dict(),
                    'optimizerE_state_dict':self.optimizerE.state_dict(),
                    'schedulerE_state_dict':self.schedulerE.state_dict(),
                    'loss_G':self.running_loss_G,
                    'loss_D':self.running_loss_D,
                },path)

    
    def loadckp(self) :
        self.ckp_load = torch.load(self.params['solver']['ckp_path'])
        self.start_epochs = self.ckp_load['epoch']
        self.netG.load_state_dict(self.ckp_load['modelG_state_dict'])
        self.netD.load_state_dict(self.ckp_load['modelD_state_dict'])
        self.netE.load_state_dict(self.ckp_load['modelE_state_dict'])
        self.optimizerG.load_state_dict(self.ckp_load['optimizerG_state_dict'])
        self.optimizerD.load_state_dict(self.ckp_load['optimizerD_state_dict'])
        self.optimizerE.load_state_dict(self.ckp_load['optimizerE_state_dict'])
        self.schedulerG.load_state_dict(self.ckp_load['schedulerG_state_dict'])
        self.schedulerD.load_state_dict(self.ckp_load['schedulerD_state_dict'])
        self.schedulerE.load_state_dict(self.ckp_load['schedulerE_state_dict'])
        lossG_load = self.ckp_load['loss_G']
        lossD_load = self.ckp_load['loss_D']

        if self.use_pool :
            self.pool = self.ckp_load['pool']

        print('Resumed training - epoch '+str(self.start_epochs+1)+' with G loss = '\
            +str(lossG_load)+' and D loss = '+str(lossD_load))

    def fitGE(self,inputG) :
        # generate some z from N(0,I)
        z = torch.randn(inputG.size(0),self.netE.nc).to(self.device)
        z_img = z.view(z.size(0),z.size(1),1,1).\
            expand(z.size(0),z.size(1),inputG.size(2),inputG.size(3))

        # generated image for cVAE-GAN
        output_vae_orig = self.netG(torch.cat([inputG,z_img],1))

        # reconstructed latent vector
        z1 = self.netE(torch.cat([output_vae_orig,inputG],1))

        # encoder loss - cVAE-GAN
        loss_latent = self.cycle_loss(z,z1)

        for _p in self.netD.parameters() :
            _p.requires_grad_(False)

        # GAN for cVAE-GAN
        if self.middle :
            pred_fake_vae1,pred_fake_vae2,pred_fake_vae3 = self.netD(output_vae_orig)
            loss_disc_vae1 = self.gan_loss(pred_fake_vae1,True)
            loss_disc_vae2 = self.gan_loss(pred_fake_vae2,True)
            loss_disc_vae3 = self.gan_loss(pred_fake_vae3,True)
            loss_disc_vae = loss_disc_vae1+self.alpha*loss_disc_vae2+self.alpha*loss_disc_vae3
        else :
            pred_fake_vae1,pred_fake_vae3 = self.netD(output_vae_orig)
            loss_disc_vae1 = self.gan_loss(pred_fake_vae1,True)
            loss_disc_vae3 = self.gan_loss(pred_fake_vae3,True)
            loss_disc_vae = loss_disc_vae1+self.alpha*loss_disc_vae3

        loss_hvs = HVSloss(output_vae_orig,inputG,self.hvs)

        loss_G = self.lambda_latent*loss_latent+loss_disc_vae+self.lambda_hvs*loss_hvs
        
        # generator weight update
        # for the generator, all the loss terms are used
        self.optimizerG.zero_grad()
        self.optimizerE.zero_grad()

        loss_G.backward()

        torch.nn.utils.clip_grad_norm_(self.netG.parameters(),1.0)
        torch.nn.utils.clip_grad_norm_(self.netE.parameters(),1.0)

        # backpropagation for generator and encoder
        self.optimizerG.step()
        self.optimizerE.step()
        # check only the L1 loss with GT colorization for the fitting procedure
        self.running_loss_G += loss_hvs.item()/self.num_batches

        return output_vae_orig, loss_hvs.item(), loss_latent.item()

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
    
    def fitD(self,imgsRGB,output_vae,epoch) :
        # enable the discriminator weights to be updated
        for _p in self.netD.parameters() :
            _p.requires_grad_(True)
        
        # discriminator weight update
        # for the discriminator only adversarial loss is needed
        self.optimizerD.zero_grad()
        if self.middle : 
            pred_real1,pred_real2,pred_real3 = self.netD(imgsRGB,noise=0.2)
            pred_fake1,pred_fake2,pred_fake3 = self.netD(output_vae,noise=0.2)

            loss_real1 = self.gan_loss(pred_real1,True)
            loss_real2 = self.gan_loss(pred_real2,True)
            loss_real3 = self.gan_loss(pred_real3,True)

            loss_fake1 = self.gan_loss(pred_fake1,False)
            loss_fake2 = self.gan_loss(pred_fake2,False)
            loss_fake3 = self.gan_loss(pred_fake3,False)

            loss_real = loss_real1+self.alpha*loss_real2+self.alpha*loss_real3
            loss_fake = loss_fake1+self.alpha*loss_fake2+self.alpha*loss_fake3
        else :
            pred_real1,pred_real3 = self.netD(imgsRGB)
            pred_fake1,pred_fake3 = self.netD(output_vae)

            loss_real1 = self.gan_loss(pred_real1,True)
            loss_real3 = self.gan_loss(pred_real3,True)

            loss_fake1 = self.gan_loss(pred_fake1,False)
            loss_fake3 = self.gan_loss(pred_fake3,False)

            loss_real = loss_real1+self.alpha*loss_real3
            loss_fake = loss_fake1+self.alpha*loss_fake3

        loss_D = loss_real+loss_fake

        # backpropagation for the discriminator
        loss_D.backward()
        torch.nn.utils.clip_grad_norm_(self.netD.parameters(),1.0)
        # check for headstart generator learning
        if epoch >= self.head_start :
            self.optimizerD.step()

        # checking adversarial loss for the fitting procedure
        self.running_loss_D += (loss_D.item())/2/self.num_batches

        return loss_D.item()

    