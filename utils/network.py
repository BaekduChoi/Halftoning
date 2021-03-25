import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'.'))

from torch import nn
import torch
from torch.nn import functional as F
from torch.nn.utils import spectral_norm as SN

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
    Pyramid Discriminator v2 class
    For each level feature extractors are shared before last convolution
"""
class _PyramidDisc2(nn.Module) :
    def __init__(self,in_ch=1,middle=True) :
        super().__init__()
        ndf = 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch,ndf,kernel_size=1,stride=1),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf,ndf*2,kernel_size=1,stride=1,bias=False),
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
            nn.Dropout(),
            nn.Conv2d(ndf*8,ndf*16,kernel_size=4,stride=2,padding=1,bias=False,padding_mode='circular'),
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(0.2,True),
            nn.Dropout()
        )
        self.out1 = nn.Conv2d(ndf*4,1,kernel_size=4,stride=1,padding=1,padding_mode='circular')
        self.out2 = nn.Conv2d(ndf*16,1,kernel_size=4,stride=1,padding=1,padding_mode='circular')
    
    def forward(self,x,noise=0.0) :
        x1 = self.conv1(x+noise*torch.randn_like(x))
        x2 = self.conv2(x1)

        return self.out1(x1),self.out2(x2)

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
    
    def forward(self,x,noise=0.) :
        return self.fc(x+noise*torch.randn_like(x))

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
    def __init__(self,in_ch=1,out_nch=1,ndf=16,ksize=3) :
        super().__init__()

        self.in_ch = in_ch
        self.out_nch = out_nch
        self.ndf = ndf
        self.norm = 'in'

        padding = (ksize-1)//2

        self.inblock = nn.Sequential(
            nn.Conv2d(in_ch,self.ndf,1),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,2*self.ndf,3,2,1,padding_mode='circular'),
            nn.ReLU(True)
        )
        self.block1 = ConvBlock(2*self.ndf,2*self.ndf,norm=self.norm,act='relu',ksize=ksize)
        self.block2 = ConvBlock(2*self.ndf,2*self.ndf,norm=self.norm,act='relu',ksize=ksize)
        self.block3 = ConvBlock(2*self.ndf,2*self.ndf,norm=self.norm,act='relu',ksize=ksize)
        # self.block4 = ConvBlock(2*self.ndf,2*self.ndf,norm='in',act='relu')

        self.blockout = nn.Sequential(
            nn.ConvTranspose2d(2*self.ndf,self.ndf,kernel_size=2,stride=2),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.out_nch,1)
        )

    def forward(self,x) :
        x1 = self.inblock(x)
        x2 = self.block1(x1)
        x3 = self.block2(x2)
        x4 = self.block3(x3)
        # x5 = self.block4(x4)

        return self.blockout(x4)


class UnetIN(nn.Module) :
    def __init__(self,in_ch=1,out_nch=1) :
        super().__init__()

        norm = 'in'

        self.in_ch = in_ch
        self.out_nch = out_nch

        self.ndf = 12

        self.inblock = nn.Sequential(
            nn.Conv2d(in_ch,self.ndf,1,1),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,3,1,1,padding_mode='circular'),
            nn.ReLU(True)
        )

        self.dsblock1 = DSConvBlock(self.ndf,self.ndf*2,norm=norm,act='relu',ds='mp')
        self.dsblock2 = DSConvBlock(self.ndf*2,self.ndf*4,norm=norm,act='relu',ds='mp')
        self.dsblock3 = DSConvBlock(self.ndf*4,self.ndf*8,norm=norm,act='relu',ds='mp')
        self.dsblock4 = DSConvBlock(self.ndf*8,self.ndf*16,norm=norm,act='relu',ds='mp')

        self.block1 = ConvBlock(self.ndf*16,self.ndf*16,norm=norm,act='relu')
        self.block2 = ConvBlock(self.ndf*16,self.ndf*16,norm=norm,act='relu')
        self.block3 = ConvBlock(self.ndf*16,self.ndf*16,norm=norm,act='relu')

        self.usblock1 = USConvBlock(self.ndf*16,self.ndf*8,norm=norm,act='relu')
        self.usblock2 = USConvBlock(self.ndf*16,self.ndf*4,norm=norm,act='relu')
        self.usblock3 = USConvBlock(self.ndf*8,self.ndf*2,norm=norm,act='relu')
        self.usblock4 = USConvBlock(self.ndf*4,self.ndf,norm=norm,act='relu')

        self.blockout = nn.Sequential(
            nn.Conv2d(self.ndf*2,self.ndf,3,1,1,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,out_nch,1)
        )

    def forward(self,x) :

        x0 = self.inblock(x)
        x1 = self.dsblock1(x0)
        x2 = self.dsblock2(x1)
        x3 = self.dsblock3(x2)
        x4 = self.dsblock4(x3)

        x4 = self.block1(x4)
        x4 = self.block2(x4)
        x4 = self.block3(x4)

        out = self.usblock1(x4)
        out = self.usblock2(torch.cat([out,x3],dim=1))
        out = self.usblock3(torch.cat([out,x2],dim=1))
        out = self.usblock4(torch.cat([out,x1],dim=1))

        return self.blockout(torch.cat([out,x0],dim=1))


class StyleNetAttn(nn.Module) :
    def __init__(self,in_ch=1,out_nch=1,ndf=16,norm='in') :
        super().__init__()

        self.in_ch = in_ch
        self.out_nch = out_nch
        self.ndf = ndf

        self.inblock = nn.Sequential(
            SN(nn.Conv2d(in_ch,self.ndf,1)),
            nn.ReLU(True),
            SN(nn.Conv2d(self.ndf,self.ndf,3,1,1,padding_mode='circular')),
            nn.ReLU(True),
            SN(nn.Conv2d(self.ndf,2*self.ndf,3,2,1,padding_mode='circular')),
            nn.ReLU(True)
        )
        self.block1 = SNConvBlock(2*self.ndf,2*self.ndf,norm=None,act='relu')
        self.block2 = SNConvBlock(2*self.ndf,2*self.ndf,norm=None,act='relu')
        self.attnblock = Attention(2*self.ndf)
        self.block3 = SNConvBlock(2*self.ndf,2*self.ndf,norm=None,act='relu')
        self.block4 = SNConvBlock(2*self.ndf,2*self.ndf,norm=None,act='relu')

        self.blockout = nn.Sequential(
            SN(nn.ConvTranspose2d(2*self.ndf,self.ndf,kernel_size=2,stride=2)),
            nn.ReLU(True),
            SN(nn.Conv2d(self.ndf,self.ndf,3,1,1,padding_mode='circular')),
            nn.ReLU(True),
            SN(nn.Conv2d(self.ndf,self.ndf,3,1,1,padding_mode='circular')),
            nn.ReLU(True),
            SN(nn.Conv2d(self.ndf,self.out_nch,1)),
            nn.Sigmoid()
        )

    def forward(self,x) :
        x1 = self.inblock(x)
        x1 = self.block1(x1)
        x1 = self.block2(x1)
        x1 = self.attnblock(x1)
        x1 = self.block3(x1)
        x1 = self.block4(x1)

        return self.blockout(x1)


"""
    Style transfer network inspired from Johnson et al with latent input
"""
class StyleNetLatent(nn.Module) :
    def __init__(self,in_ch=1,out_nch=1,ndf=16,ksize=3,latent_dim=16) :
        super().__init__()

        self.in_ch = in_ch
        self.out_nch = out_nch
        self.ndf = ndf
        self.norm = 'in'
        self.latent_dim = latent_dim

        padding = (ksize-1)//2

        self.inblock = nn.Sequential(
            nn.Conv2d(in_ch,self.ndf,1),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,2*self.ndf,3,2,1,padding_mode='circular'),
            nn.ReLU(True)
        )
        self.block1 = ConvBlock(2*self.ndf,2*self.ndf,norm=self.norm,act='relu',ksize=ksize)
        self.block2 = ConvBlock(2*self.ndf,2*self.ndf,norm=self.norm,act='relu',ksize=ksize)
        self.block3 = ConvBlock(2*self.ndf,2*self.ndf,norm=self.norm,act='relu',ksize=ksize)

        self.blockout = nn.Sequential(
            nn.ConvTranspose2d(2*self.ndf,self.ndf,kernel_size=2,stride=2),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.out_nch,1)
        )

        self.g1 = nn.Parameter(0.02*torch.randn(1))
        self.g2 = nn.Parameter(0.02*torch.randn(1))
        self.g3 = nn.Parameter(0.02*torch.randn(1))

    def forward(self,x,z) :

        z1 = z.view(-1,1,self.latent_dim,self.latent_dim)
        
        x1 = self.inblock(x)

        target_size = [x1.size(2),x1.size(3)]
        z1 = F.interpolate(z1,target_size,mode='bilinear')

        x2 = self.block1(x1+self.g1*z1)
        x3 = self.block2(x2+self.g2*z1)
        x4 = self.block3(x3+self.g3*z1)

        return self.blockout(x4)


"""
    Style transfer network inspired from Johnson et al with noise injection
"""
class StyleNetNoise(nn.Module) :
    def __init__(self,in_ch=1,out_nch=1,ndf=16,ksize=3) :
        super().__init__()

        self.in_ch = in_ch
        self.out_nch = out_nch
        self.ndf = ndf
        self.norm = 'in'

        padding = (ksize-1)//2

        self.inblock = nn.Sequential(
            nn.Conv2d(in_ch,self.ndf,1),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,2*self.ndf,3,2,1,padding_mode='circular'),
            nn.ReLU(True)
        )
        self.block1 = ConvBlock(2*self.ndf,2*self.ndf,norm=self.norm,act='relu',ksize=ksize)
        self.block2 = ConvBlock(2*self.ndf,2*self.ndf,norm=self.norm,act='relu',ksize=ksize)
        self.block3 = ConvBlock(2*self.ndf,2*self.ndf,norm=self.norm,act='relu',ksize=ksize)

        self.blockout = nn.Sequential(
            nn.ConvTranspose2d(2*self.ndf,self.ndf,kernel_size=2,stride=2),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.out_nch,1)
        )

        self.g1 = nn.Parameter(0.02*torch.randn(1))
        self.g2 = nn.Parameter(0.02*torch.randn(1))
        self.g3 = nn.Parameter(0.02*torch.randn(1))
        self.g4 = nn.Parameter(0.02*torch.randn(1))

    def forward(self,x) :
        
        x1 = self.inblock(x)
        x2 = self.block1(x1+self.g1*torch.randn_like(x1))
        x3 = self.block2(x2+self.g2*torch.randn_like(x2))
        x4 = self.block3(x3+self.g3*torch.randn_like(x3))

        return self.blockout(x4+self.g4*torch.randn_like(x4))


"""
    Style transfer network inspired from Johnson et al with noise injection
"""
class StyleNetCIN(nn.Module) :
    def __init__(self,in_ch=1,out_nch=1,ndf=16,ksize=3,latent_dim=16) :
        super().__init__()

        self.in_ch = in_ch
        self.out_nch = out_nch
        self.ndf = ndf
        self.norm = 'cin'

        padding = (ksize-1)//2

        self.inblock = nn.Sequential(
            nn.Conv2d(in_ch,self.ndf,1),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,2*self.ndf,3,2,1,padding_mode='circular'),
            nn.ReLU(True)
        )
        self.block1 = ConvBlock(2*self.ndf,2*self.ndf,norm=self.norm,act='relu',ksize=ksize)
        self.block2 = ConvBlock(2*self.ndf,2*self.ndf,norm=self.norm,act='relu',ksize=ksize)
        self.block3 = ConvBlock(2*self.ndf,2*self.ndf,norm=self.norm,act='relu',ksize=ksize)

        self.blockout = nn.Sequential(
            nn.ConvTranspose2d(2*self.ndf,self.ndf,kernel_size=2,stride=2),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.out_nch,1)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim,6*self.ndf)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(latent_dim,6*self.ndf)
        )

    def forward(self,x,z) :

        g = self.fc1(z)
        b = self.fc2(z)
        
        x1 = self.inblock(x)
        x2 = self.block1(x1,g[:,:2*self.ndf],b[:,:2*self.ndf])
        x3 = self.block2(x2,g[:,2*self.ndf:4*self.ndf],b[:,2*self.ndf:4*self.ndf])
        x4 = self.block3(x3,g[:,4*self.ndf:6*self.ndf],b[:,4*self.ndf:6*self.ndf])

        return self.blockout(x4)

"""
    Style transfer network inspired from Johnson et al with noise injection
"""
class StyleNetAppend(nn.Module) :
    def __init__(self,in_ch=1,out_nch=1,ndf=16,ksize=3,latent_dim=16) :
        super().__init__()

        self.in_ch = in_ch
        self.out_nch = out_nch
        self.ndf = ndf
        self.norm = 'in'

        padding = (ksize-1)//2

        self.inblock = nn.Sequential(
            nn.Conv2d(in_ch+latent_dim,self.ndf,1),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,2*self.ndf,3,2,1,padding_mode='circular'),
            nn.ReLU(True)
        )
        self.block1 = ConvBlock(2*self.ndf,2*self.ndf,norm=self.norm,act='relu',ksize=ksize)
        self.block2 = ConvBlock(2*self.ndf,2*self.ndf,norm=self.norm,act='relu',ksize=ksize)
        self.block3 = ConvBlock(2*self.ndf,2*self.ndf,norm=self.norm,act='relu',ksize=ksize)

        self.blockout = nn.Sequential(
            nn.ConvTranspose2d(2*self.ndf,self.ndf,kernel_size=2,stride=2),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.out_nch,1)
        )

    def forward(self,x,z) :

        z_vae = z.view(z.size(0),z.size(1),1,1).expand(z.size(0),z.size(1),x.size(2),x.size(3))
        x0 = torch.cat([x,z_vae],1)
        
        x1 = self.inblock(x0)
        x2 = self.block1(x1)
        x3 = self.block2(x2)
        x4 = self.block3(x3)

        return self.blockout(x4)