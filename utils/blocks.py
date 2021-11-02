from torch import nn
import torch
from torch.nn import functional as F
from torch.nn.utils import spectral_norm as SN

"""
    Conditional instance normalization layer
"""
class ConditionalInstanceNorm(nn.Module) :
    def __init__(self,num_ch) :
        super().__init__()
        self.num_ch = num_ch

    def forward(self,x,gamma,beta) :
        x = F.instance_norm(x)

        g = gamma.view(x.size(0),self.num_ch,1,1)
        b = beta.view(x.size(0),self.num_ch,1,1)

        return x*g+b

"""
    Conv block
"""
class ConvBlock(nn.Module) :
    def __init__(self,in_ch,out_ch,cat=False,norm='bn',act='relu',ksize=3) :
        super().__init__()

        self.cat = cat
        if self.cat and in_ch != out_ch :
            raise ValueError('channel numbers for input and output do not match')

        if norm == 'bn' :
            self.norm = nn.BatchNorm2d(out_ch)
        elif norm == 'in' :
            self.norm = nn.InstanceNorm2d(out_ch,affine=True)
        elif norm == 'cin' :
            self.norm = ConditionalInstanceNorm(out_ch)
        else :
            self.norm = None
        self.normtype = norm

        padding = (ksize-1)//2
        
        if act == 'lrelu' :
            self.act = nn.LeakyReLU(0.2,True)
        else : # default = ReLU
            self.act = nn.ReLU(True)
            
        self.conv1 = nn.Conv2d(in_ch,out_ch,kernel_size=ksize,padding=padding,padding_mode='circular')
        self.conv2 = nn.Conv2d(out_ch,out_ch,kernel_size=ksize,padding=padding,padding_mode='circular')
    
    def forward(self,x,g=None,b=None) :
        x1 = self.conv1(x)
        x1 = self.act(x1)
        x1 = self.conv2(x1)
        x1 = self.act(x1)

        if self.norm is not None :
            x1 = self.norm(x1,g,b) if self.normtype == 'cin' else self.norm(x1)

        out = torch.cat([x,x1],dim=1) if self.cat else x1

        return out

class ConvBlockINEDense(nn.Module) :
    def __init__(self,n_ch,act='relu',ksize=3,norm='in',padding_mode='circular') :
        super().__init__()

        padding = (ksize-1)//2
        
        if act == 'lrelu' :
            self.act = nn.LeakyReLU(0.2,True)
        else : # default = ReLU
            self.act = nn.ReLU(True)
            
        self.conv1 = nn.Conv2d(n_ch,n_ch,kernel_size=ksize,padding=padding,padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(2*n_ch,n_ch,kernel_size=ksize,padding=padding,padding_mode=padding_mode)
        self.conv3 = nn.Conv2d(3*n_ch,n_ch,kernel_size=ksize,padding=padding,padding_mode=padding_mode)
        self.conv4 = nn.Conv2d(4*n_ch,n_ch,kernel_size=ksize,padding=padding,padding_mode=padding_mode)
        
        self.norm = norm
        if norm == 'in' :
            self.norm1 = nn.InstanceNorm2d(n_ch,affine=True)
            self.norm2 = nn.InstanceNorm2d(n_ch,affine=True)
            self.norm3 = nn.InstanceNorm2d(n_ch,affine=True)
    
    def forward(self,x,g=None,b=None) :
        x1 = self.conv1(x)
        x1 = self.act(x1)
        if self.norm == 'in' :
            x1 = self.norm1(x1)

        x2 = torch.cat([x1,x],dim=1)
        x2 = self.conv2(x2)
        x2 = self.act(x2)
        if self.norm == 'in' :
            x2 = self.norm2(x2)

        x3 = torch.cat([x2,x1,x],dim=1)
        x3 = self.conv3(x3)
        x3 = self.act(x3)
        if self.norm == 'in' :
            x3 = self.norm3(x3)

        x4 = torch.cat([x3,x2,x1,x],dim=1)
        out = self.conv4(x4)

        return out