import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'.'))

from torch import nn
import torch
from torch.nn import functional as F

from blocks import *

def klvloss(mu,logvar) :
    return torch.mean(-0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp()))

"""
    Pyramid Discriminator class
    For each level feature extractors are shared before last convolution
"""
class PyramidDisc(nn.Module) :
    def __init__(self,in_ch=1,middle=True) :
        super().__init__()
        ndf = 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch,ndf,kernel_size=1,stride=1),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1,bias=False,padding_mode='circular'),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1,bias=False,padding_mode='circular'),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2,True),
            nn.Dropout()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf*4,ndf*8,kernel_size=4,stride=2,padding=1,bias=False,padding_mode='circular'),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2,True),
            nn.Dropout()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf*8,ndf*16,kernel_size=4,stride=2,padding=1,bias=False,padding_mode='circular'),
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(0.2,True),
            nn.Dropout()
        )
        self.out1 = nn.Conv2d(ndf*4,1,kernel_size=1,stride=1)
        self.middle = middle
        if self.middle :
            self.out2 = nn.Conv2d(ndf*8,1,kernel_size=1,stride=1)
        self.out3 = nn.Conv2d(ndf*16,1,kernel_size=1,stride=1)
    
    def forward(self,x,noise=0.0) :
        x1 = self.conv1(x+noise*torch.randn_like(x))
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        if self.middle :
            return self.out1(x1),self.out2(x2),self.out3(x3)
        else :
            return self.out1(x1),self.out3(x3)

"""
    Encoder for VAE-GAN based implementation
"""
class EncoderWB(nn.Module) :
    def __init__(self,in_ch=3,out_nc=16,input_size=256,vae=False,norm='in') :
        super().__init__()
        self.nc = out_nc
        ndf = 16
        self.block = nn.Sequential(
            ConvBlock(in_ch,ndf,norm=None,act='lrelu'),
            DSConvBlock(ndf,ndf*2,norm=norm,act='lrelu'),
            DSConvBlock(ndf*2,ndf*4,norm=norm,act='lrelu'),
            DSConvBlock(ndf*4,ndf*8,norm=norm,act='lrelu'),
            DSConvBlock(ndf*8,ndf*16,norm=norm,act='lrelu')
        )

        self.fc1 = nn.Linear(((input_size//(2**4))**2)*ndf*16,out_nc)
        if vae :
            self.fc2 = nn.Linear(((input_size//(2**4))**2)*ndf*16,out_nc)
        self.vae = vae
    
    def forward(self,x) :
        xc = self.block(x)
        xv = xc.view(xc.size(0),-1)
        if self.vae :
            out1 = self.fc1(xv)
            out2 = self.fc2(xv)
            return out1,out2
        else :
            out = self.fc1(xv)
            return out

"""
    Encoder for VAE-GAN based implementation
"""
class EncoderWB2(nn.Module) :
    def __init__(self,in_ch=3,out_nc=16,input_size=256,vae=False,norm='in',ksize=3) :
        super().__init__()
        self.nc = out_nc
        ndf = 16
        self.block = nn.Sequential(
            ConvBlock(in_ch,ndf,norm=None,act='lrelu',ksize=ksize),
            DSConvBlock(ndf,ndf*2,norm=norm,act='lrelu',ksize=ksize),
            ConvBlock(ndf*2,ndf*2,norm=norm,act='lrelu',ksize=ksize),
            DSConvBlock(ndf*2,ndf*4,norm=norm,act='lrelu',ksize=ksize),
            ConvBlock(ndf*4,ndf*4,norm=norm,act='lrelu',ksize=ksize),
            DSConvBlock(ndf*4,ndf*8,norm=norm,act='lrelu',ksize=ksize),
            ConvBlock(ndf*8,ndf*8,norm=norm,act='lrelu',ksize=ksize),
            DSConvBlock(ndf*8,ndf*16,norm=norm,act='lrelu',ksize=ksize),
            ConvBlock(ndf*16,ndf*16,norm=norm,act='lrelu',ksize=ksize)
        )

        self.fc1 = nn.Linear(((input_size//(2**4))**2)*ndf*16,out_nc)
        if vae :
            self.fc2 = nn.Linear(((input_size//(2**4))**2)*ndf*16,out_nc)
        self.vae = vae
    
    def forward(self,x) :
        xc = self.block(x)
        xv = xc.view(xc.size(0),-1)
        if self.vae :
            out1 = self.fc1(xv)
            out2 = self.fc2(xv)
            return out1,out2
        else :
            out = self.fc1(xv)
            return out
    
class EncoderWB3(nn.Module) :
    def __init__(self,in_ch=3,out_nc=16,input_size=256,vae=False,norm='in',ksize=3) :
        super().__init__()
        self.nc = out_nc
        ndf = 16
        self.block = nn.Sequential(
            ConvBlock(in_ch,ndf,norm=None,act='lrelu',ksize=ksize),
            ConvBlock(ndf,ndf,norm=norm,act='lrelu',ksize=ksize),
            DSConvBlock(ndf,ndf*2,norm=norm,act='lrelu',ksize=ksize),
            ConvBlock(ndf*2,ndf*2,norm=norm,act='lrelu',ksize=ksize),
            ConvBlock(ndf*2,ndf*2,norm=norm,act='lrelu',ksize=ksize),
            DSConvBlock(ndf*2,ndf*4,norm=norm,act='lrelu',ksize=ksize),
            ConvBlock(ndf*4,ndf*4,norm=norm,act='lrelu',ksize=ksize),
            ConvBlock(ndf*4,ndf*4,norm=norm,act='lrelu',ksize=ksize),
            DSConvBlock(ndf*4,ndf*8,norm=norm,act='lrelu',ksize=ksize),
            ConvBlock(ndf*8,ndf*8,norm=norm,act='lrelu',ksize=ksize),
            ConvBlock(ndf*8,ndf*8,norm=norm,act='lrelu',ksize=ksize),
            DSConvBlock(ndf*8,ndf*16,norm=norm,act='lrelu',ksize=ksize),
            ConvBlock(ndf*16,ndf*16,norm=norm,act='lrelu',ksize=ksize),
            ConvBlock(ndf*16,ndf*16,norm=norm,act='lrelu',ksize=ksize),
        )

        self.fc1 = nn.Linear(((input_size//(2**4))**2)*ndf*16,out_nc)
        if vae :
            self.fc2 = nn.Linear(((input_size//(2**4))**2)*ndf*16,out_nc)
        self.vae = vae
    
    def forward(self,x) :
        xc = self.block(x)
        xv = xc.view(xc.size(0),-1)
        if self.vae :
            out1 = self.fc1(xv)
            out2 = self.fc2(xv)
            return out1,out2
        else :
            out = self.fc1(xv)
            return out

"""
    Discriminator for GAN-based implementation
    Based on the PatchGAN structure in https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    Added Dropout for slower discriminator fitting
"""
class Discriminator(nn.Module) :
    def __init__(self,input_ch=1,ndf=64) :
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(input_ch,ndf,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf*4,ndf*8,kernel_size=4,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf*8,1,kernel_size=4,stride=1,padding=1)
        )
    
    def forward(self,x,noise=0.0) :
        return self.block(x+noise*torch.randn_like(x))

"""
    Discriminator for GAN-based implementation
    Added Dropout for slower discriminator fitting
"""
class Discriminator2(nn.Module) :
    def __init__(self,in_ch=1,ndf=64) :
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,ndf,kernel_size=1,stride=1),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf,ndf,kernel_size=2,stride=2,bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf,ndf*2,kernel_size=2,stride=2,bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf*2,1,kernel_size=1,stride=1)
        )
    
    def forward(self,x,noise=0.0) :
        return self.block(x,noise*torch.randn_like(x))

class DiscAAE(nn.Module) :
    def __init__(self,in_dim=16) :
        super().__init__()
        self.nc = in_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim,64),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(0.5),
            nn.Linear(64,1)
        )
    
    def forward(self,x) :
        return self.fc(x)

"""
    Loss for LSGAN
"""
class LSGANLoss(nn.Module) :
    def __init__(self,device) :
        super().__init__()
        self.loss = nn.MSELoss()
        self.device = device
        
    def get_label(self,prediction,is_real) :
        if is_real : 
            return torch.ones_like(prediction)
        else :
            return torch.zeros_like(prediction)
    
    def __call__(self,prediction,is_real) :
        label = self.get_label(prediction,is_real)
        label.to(self.device)
        return self.loss(prediction,label)

"""
    Style transfer network inspired from Johnson et al
    Replaced BN to IN everywhere, added Conv in front, replaced ResBlock with ConvBlock
"""
class StyleNetIN(nn.Module) :
    def __init__(self,in_ch=1,out_nch=1,ndf=16) :
        super().__init__()

        self.in_ch = in_ch
        self.out_nch = out_nch
        self.ndf = ndf

        self.inblock = nn.Sequential(
            nn.Conv2d(in_ch,self.ndf,1),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,3,1,1,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,2*self.ndf,3,2,1,padding_mode='circular'),
            nn.InstanceNorm2d(2*self.ndf),
            nn.ReLU(True)
        )
        self.block1 = ConvBlock(2*self.ndf,2*self.ndf,norm='in',act='relu',ksize=3)
        self.block2 = ConvBlock(2*self.ndf,2*self.ndf,norm='in',act='relu',ksize=3)
        self.block3 = ConvBlock(2*self.ndf,2*self.ndf,norm='in',act='relu',ksize=3)
        self.block4 = ConvBlock(2*self.ndf,2*self.ndf,norm='in',act='relu',ksize=3)

        self.blockout = nn.Sequential(
            nn.ConvTranspose2d(2*self.ndf,self.ndf,kernel_size=2,stride=2),
            nn.InstanceNorm2d(self.ndf),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,3,1,1,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.out_nch,1)
        )

    def forward(self,x) :
        x1 = self.inblock(x)
        x2 = self.block1(x1)
        x3 = self.block2(x2)
        x4 = self.block3(x3)
        x5 = self.block4(x4)

        return self.blockout(x5)