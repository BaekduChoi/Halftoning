import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'.'))

from torch import nn
import torch
from torch.nn import functional as F
from torch.nn import init as init
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

class SimpleNetDenseINDepth3(nn.Module) :
    def __init__(self,in_ch=1,out_nch=1,ndf=32,ksize=7,depth=6) :
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

        L1 = [ConvBlockINEDense(self.ndf,act='relu',ksize=ksize) for i in range(depth-2)]
        self.block1 = nn.Sequential(*L1)

        L2 = [ConvBlockINEDense(self.ndf,act='relu',ksize=ksize) for i in range(2)]
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

class SimpleNetDenseLN(nn.Module) :
    def __init__(self,in_ch=1,out_nch=1,ndf=32,ksize=7,depth=6) :
        super().__init__()

        self.in_ch = in_ch
        self.out_nch = out_nch
        self.ndf = ndf

        padding = (ksize-1)//2

        self.inblock = nn.Sequential(
            nn.Conv2d(in_ch,self.ndf//2,1),
            nn.ReLU(True),
            nn.Conv2d(self.ndf//2,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
        )

        L1 = [ConvBlockLNEDense(self.ndf,act='relu',ksize=ksize) for i in range(depth-2)]
        self.block1 = nn.Sequential(*L1)

        L2 = [ConvBlockLNEDense(self.ndf,act='relu',ksize=ksize) for i in range(2)]
        self.block2 = nn.Sequential(*L2)

        self.blockmid = nn.Sequential(
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf//2,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf//2,self.out_nch,1),
            nn.Sigmoid()
        )

        self.blockout = nn.Sequential(
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf//2,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf//2,self.out_nch,1),
            nn.Sigmoid()
        )

    def forward(self,x) :

        x1 = self.inblock(x)
        x2 = self.block1(x1)
        x3 = self.block2(x2)

        return self.blockout(x3), self.blockmid(x2)

def init_weights_small(m) :
    L = list(m.modules())

    for layer in L :
        if isinstance(layer,nn.Sequential) :
            for idx in range(len(layer)) :
                init_weights_small(layer[idx])
        elif isinstance(layer,nn.Conv2d) :
            layer.weight.data.normal_(0.0,0.001)
            layer.bias.data.fill_(0)
        elif isinstance(layer,DenseWSAN) :
            init_weights_small(layer.block)

def init_weights_kaiming(m) :
    L = list(m.modules())

    for layer in L :
        if isinstance(layer,nn.Sequential) :
            for idx in range(len(layer)) :
                init_weights_kaiming(layer[idx])
        elif isinstance(layer,nn.Conv2d) :
            init.kaiming_normal_(layer.weight,mode='fan_out',nonlinearity='relu')
            layer.bias.data.fill_(0)
        elif isinstance(layer,DenseWSAN) :
            init_weights_kaiming(layer.block)
"""
    Using spatially-adaptive conditional normalization 
    similar to that in SPADE (https://github.com/NVlabs/SPADE/blob/master/)
"""
class SimpleNetDenseINDepthAdaptive(nn.Module) :
    def __init__(self,out_nch=1,ndf=32,ksize=7) :
        super().__init__()

        self.out_nch = out_nch
        self.ndf = ndf

        padding = (ksize-1)//2

        self.inblock = nn.Sequential(
            nn.Conv2d(1,self.ndf,1),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True)
        )
        
        self.paramblock1 = nn.Sequential(
            nn.Conv2d(1,self.ndf,1),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,2*self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(2*self.ndf,2*self.ndf,1),
        )

        self.paramblock2 = nn.Sequential(
            nn.Conv2d(1,self.ndf,1),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,2*self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(2*self.ndf,2*self.ndf,1),
        )

        self.paramblock3 = nn.Sequential(
            nn.Conv2d(1,self.ndf,1),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,2*self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(2*self.ndf,2*self.ndf,1),
        )

        self.paramblock4 = nn.Sequential(
            nn.Conv2d(1,self.ndf,1),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,2*self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(2*self.ndf,2*self.ndf,1),
        )

        self.paramblock5 = nn.Sequential(
            nn.Conv2d(1,self.ndf,1),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,2*self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(2*self.ndf,2*self.ndf,1),
        )

        self.paramblock6 = nn.Sequential(
            nn.Conv2d(1,self.ndf,1),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,2*self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(2*self.ndf,2*self.ndf,1),
        )

        self.block1 = DenseWSAN(ConvBlockINEDense(self.ndf,act='relu',ksize=ksize),n_ch=self.ndf)
        self.block2 = DenseWSAN(ConvBlockINEDense(self.ndf,act='relu',ksize=ksize),n_ch=self.ndf)
        self.block3 = DenseWSAN(ConvBlockINEDense(self.ndf,act='relu',ksize=ksize),n_ch=self.ndf)
        self.block4 = DenseWSAN(ConvBlockINEDense(self.ndf,act='relu',ksize=ksize),n_ch=self.ndf)

        self.block5 = DenseWSAN(ConvBlockINEDense(self.ndf,act='relu',ksize=ksize),n_ch=self.ndf)
        self.block6 = DenseWSAN(ConvBlockINEDense(self.ndf,act='relu',ksize=ksize),n_ch=self.ndf)

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

        # self.inblock.apply(init_weights_kaiming)
        # self.block1.apply(init_weights_kaiming)
        # self.block2.apply(init_weights_kaiming)
        # self.block3.apply(init_weights_kaiming)
        # self.block4.apply(init_weights_kaiming)
        # self.block5.apply(init_weights_kaiming)
        # self.block6.apply(init_weights_kaiming)
        # self.blockmid.apply(init_weights_kaiming)
        # self.blockout.apply(init_weights_kaiming)

        # self.paramblock1.apply(init_weights_small)
        # self.paramblock2.apply(init_weights_small)
        # self.paramblock3.apply(init_weights_small)
        # self.paramblock4.apply(init_weights_small)
        # self.paramblock5.apply(init_weights_small)
        # self.paramblock6.apply(init_weights_small)

    def forward(self,img,scr) :

        # gb1 = self.paramblock1(img)
        # gb2 = self.paramblock2(img)
        # gb3 = self.paramblock3(img)
        # gb4 = self.paramblock4(img)
        # gb5 = self.paramblock5(img)
        # gb6 = self.paramblock6(img)

        # x1 = self.inblock(scr)

        gb1 = self.paramblock1(scr)
        gb2 = self.paramblock2(scr)
        gb3 = self.paramblock3(scr)
        gb4 = self.paramblock4(scr)
        gb5 = self.paramblock5(scr)
        gb6 = self.paramblock6(scr)

        x1 = self.inblock(img)

        x1 = self.block1(x1,gb1[:,:self.ndf,:,:],gb1[:,self.ndf:,:,:])
        x1 = self.block2(x1,gb2[:,:self.ndf,:,:],gb2[:,self.ndf:,:,:])
        x1 = self.block3(x1,gb3[:,:self.ndf,:,:],gb3[:,self.ndf:,:,:])
        x2 = self.block4(x1,gb4[:,:self.ndf,:,:],gb4[:,self.ndf:,:,:])

        x3 = self.block5(x2,gb5[:,:self.ndf,:,:],gb5[:,self.ndf:,:,:])
        x3 = self.block6(x3,gb6[:,:self.ndf,:,:],gb6[:,self.ndf:,:,:])

        return self.blockout(x3), self.blockmid(x2)