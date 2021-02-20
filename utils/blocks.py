from torch import nn
import torch
from torch.nn import functional as F

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

        return self.relu(x*g+b)

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
            self.norm = nn.InstanceNorm2d(out_ch)
        elif norm == 'cin' :
            self.norm = ConditionalInstanceNorm(out_ch)
        else :
            self.norm = None
        self.normtype = norm
        
        if act == 'lrelu' :
            self.act = nn.LeakyReLU(0.2,True)
        else : # default = ReLU
            self.act = nn.ReLU(True)
        
        padding = (ksize-1)//2

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

"""
    Conv block w Downscaling x2
"""
class DSConvBlock(nn.Module) :
    def __init__(self,in_ch,out_ch,norm='bn',act='relu',ds='mp',ksize=3) :
        super().__init__()

        if norm == 'bn' :
            self.norm = nn.BatchNorm2d(out_ch)
        elif norm == 'in' :
            self.norm = nn.InstanceNorm2d(out_ch)
        else :
            self.norm = None
        
        if act == 'lrelu' :
            self.act = nn.LeakyReLU(0.2,True)
        else : # default = ReLU
            self.act = nn.ReLU(True)

        padding = (ksize-1)//2

        self.ds = nn.MaxPool2d(2) if ds == 'mp' \
            else nn.Conv2d(in_ch,in_ch,kernel_size=3,stride=2,padding=1,padding_mode='circular')
        self.conv1 = nn.Conv2d(in_ch,out_ch,kernel_size=ksize,padding=padding,padding_mode='circular')
        self.conv2 = nn.Conv2d(out_ch,out_ch,kernel_size=ksize,padding=padding,padding_mode='circular')
    
    def forward(self,x,g=None,b=None) :
        x1 = self.ds(x)
        x1 = self.conv1(x1)
        x1 = self.act(x1)
        x1 = self.conv2(x1)
        x1 = self.act(x1)

        if self.norm is not None :
            out = self.norm(x1)
        else :
            out = x1

        return out

"""
    Conv block w Upscaling x2
"""
class USConvBlock(nn.Module) :
    def __init__(self,in_ch,out_ch,norm='bn',act='relu',ksize=3) :
        super().__init__()

        if norm == 'bn' :
            self.norm = nn.BatchNorm2d(out_ch)
        elif norm == 'in' :
            self.norm = nn.InstanceNorm2d(out_ch)
        else :
            self.norm = None
        
        if act == 'lrelu' :
            self.act = nn.LeakyReLU(0.2,True)
        else : # default = ReLU
            self.act = nn.ReLU(True)

        padding = (ksize-1)//2

        self.us = nn.Conv2dTranspose(in_ch,in_ch,kernel_size=2,stride=2)
        self.conv1 = nn.Conv2d(in_ch,out_ch,kernel_size=ksize,padding=padding,padding_mode='circular')
        self.conv2 = nn.Conv2d(out_ch,out_ch,kernel_size=ksize,padding=padding,padding_mode='circular')
    
    def forward(self,x,g=None,b=None) :
        x1 = self.act(self.conv1(x))
        x1 = self.act(self.conv2(x1))
        x1 = self.act(self.us(x))

        if self.norm is not None :
            out = self.norm(x1)
        else :
            out = x1

        return out

class ConvCatBlockCIN(nn.Module) :
    def __init__(self,ch_num) :
        super().__init__()
        self.conv1 = nn.Conv2d(ch_num,ch_num,3,1,1,padding_mode='circular')
        self.conv2 = nn.Conv2d(ch_num,ch_num,3,1,1,padding_mode='circular')
        self.relu = nn.ReLU(True)
    
    def forward(self,x,gamma1,beta1,gamma2,beta2) :
        x1 = self.conv1(x)
        x2 = F.instance_norm(x1)

        g1 = gamma1.view(x.size(0),-1,1,1)
        b1 = beta1.view(x.size(0),-1,1,1)

        x3 = F.instance_norm(self.conv2(self.relu(x2*g1+b1)))

        g2 = gamma2.view(x.size(0),-1,1,1)
        b2 = beta2.view(x.size(0),-1,1,1)

        return torch.cat([x,x3*g2+b2],dim=1)