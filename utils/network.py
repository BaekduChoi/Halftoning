import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'.'))

from torch import nn
import torch
from torch.nn import functional as F
from torch.nn.utils import spectral_norm as SN

from blocks import *

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
        return self.block(x+noise*torch.randn_like(x))

"""
    Discriminator for GAN-based implementation
    Based on the PatchGAN structure in https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    Added Dropout for slower discriminator fitting
"""
class Discriminator(nn.Module) :
    def __init__(self,in_ch=1,ndf=64) :
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,ndf,kernel_size=4,stride=2,padding=1,padding_mode='circular'),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1,bias=False,padding_mode='circular'),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1,bias=False,padding_mode='circular'),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf*4,ndf*8,kernel_size=4,stride=1,padding=1,bias=False,padding_mode='circular'),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf*8,1,kernel_size=4,stride=1,padding=1,padding_mode='circular')
        )
    
    def forward(self,x,noise=0.0) :
        return self.block(x+noise*torch.randn_like(x))

class SimpleNetDenseIN(nn.Module) :
    def __init__(self,in_ch=1,out_nch=1,ndf=32,ksize=7) :
        super().__init__()

        self.in_ch = in_ch
        self.out_nch = out_nch
        self.ndf = ndf

        padding = (ksize-1)//2

        self.inblock = nn.Sequential(
            nn.Conv2d(in_ch,self.ndf,1),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
        )
        self.block1 = ConvBlockINEDense(self.ndf,act='relu',ksize=ksize)
        self.block2 = ConvBlockINEDense(self.ndf,act='relu',ksize=ksize)
        self.block3 = ConvBlockINEDense(self.ndf,act='relu',ksize=ksize)
        self.block4 = ConvBlockINEDense(self.ndf,act='relu',ksize=ksize)

        self.blockout = nn.Sequential(
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.out_nch,1),
            nn.Sigmoid()
        )

    def forward(self,x) :

        x1 = self.inblock(x)
        x2 = self.block1(x1)
        x3 = self.block2(x2)
        x4 = self.block3(x3)
        x5 = self.block4(x4)

        return self.blockout(x5)

class SimpleNetDenseINDepth(nn.Module) :
    def __init__(self,in_ch=1,out_nch=1,ndf=32,ksize=7,depth=4) :
        super().__init__()

        self.in_ch = in_ch
        self.out_nch = out_nch
        self.ndf = ndf

        padding = (ksize-1)//2

        self.inblock = nn.Sequential(
            nn.Conv2d(in_ch,self.ndf,1),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
        )

        L = [ConvBlockINEDense(self.ndf,act='relu',ksize=ksize) for i in range(depth)]
        self.block = nn.Sequential(*L)

        self.blockout = nn.Sequential(
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.out_nch,1),
            nn.Sigmoid()
        )

    def forward(self,x) :

        x1 = self.inblock(x)
        x2 = self.block(x1)

        return self.blockout(x2)

class SimpleNetDenseINDepth2(nn.Module) :
    def __init__(self,in_ch=1,out_nch=1,ndf=32,ksize=7,depth=4) :
        super().__init__()

        self.in_ch = in_ch
        self.out_nch = out_nch
        self.ndf = ndf

        padding = (ksize-1)//2

        self.inblock = nn.Sequential(
            nn.Conv2d(in_ch,self.ndf,1),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
        )

        L1 = [ConvBlockINEDense(self.ndf,act='relu',ksize=ksize) for i in range(4)]
        self.block1 = nn.Sequential(*L1)

        L2 = [ConvBlockINEDense(self.ndf,act='relu',ksize=ksize) for i in range(depth-4)]
        self.block2 = nn.Sequential(*L2)

        self.blockmid = nn.Sequential(
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.out_nch,1),
            nn.Sigmoid()
        )

        self.blockout = nn.Sequential(
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.out_nch,1),
            nn.Sigmoid()
        )

    def forward(self,x) :

        x1 = self.inblock(x)
        x2 = self.block1(x1)
        x3 = self.block2(x2)

        return self.blockout(x3), self.blockmid(x2)

class SNDINwSN(nn.Module) :
    def __init__(self,in_ch=1,out_nch=1,ndf=32,ksize=7,depth=6) :
        super().__init__()

        self.in_ch = in_ch
        self.out_nch = out_nch
        self.ndf = ndf

        padding = (ksize-1)//2

        self.inblock = nn.Sequential(
            SN(nn.Conv2d(in_ch,self.ndf,1)),
            nn.LeakyReLU(0.2,True),
            SN(nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular')),
            nn.LeakyReLU(0.2,True),
        )

        L = [ConvBlockINEDenseSN(self.ndf,act='lrelu',ksize=ksize) for i in range(depth)]
        self.block = nn.Sequential(*L)

        self.blockout = nn.Sequential(
            SN(nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular')),
            nn.LeakyReLU(0.2,True),
            SN(nn.Conv2d(self.ndf,self.out_nch,1))
        )

    def forward(self,x) :

        x1 = self.inblock(x)
        x2 = self.block(x1)

        return self.blockout(x2)