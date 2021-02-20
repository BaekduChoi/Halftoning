from torch import nn
import torch
from torch.nn import functional as F

"""
    Conv block for U-net
"""
class ConvBlock(nn.Module) :
    def __init__(self,in_ch,out_ch) :
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x) :
        return self.block(x)

"""
    Downconv block for U-net
"""
class DownConvBlock(nn.Module) :
    def __init__(self,in_ch,out_ch) :
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_ch,out_ch)
        )
        
    def forward(self,x) :
        return self.block(x)

"""
    Upconv block without concatenation for U-net
"""
class UpConvBlockNoConcat(nn.Module) :
    def __init__(self,in_ch,out_ch) :
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_ch,in_ch),
            nn.ConvTranspose2d(in_ch,out_ch,kernel_size=2,stride=2)
        )
        
    def forward(self,x) :
        return self.block(x)
    
"""
    Upconv block with concat for U-net
"""
class UpConvBlock(nn.Module) :
    def __init__(self,in_ch,out_ch) :
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(2*in_ch,in_ch),
            nn.ConvTranspose2d(in_ch,out_ch,kernel_size=2,stride=2)
        )
        
    def forward(self,x1,x2) :
        x = torch.cat([x1,x2],dim=1)
        return self.block(x)
    
"""
    Output conv block with concat for U-net
"""
class OutBlock(nn.Module) :
    def __init__(self,in_ch,out_ch) :
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(2*in_ch,in_ch),
            nn.Conv2d(in_ch,out_ch,kernel_size=1)
        )
    
    def forward(self,x1,x2) :
        x = torch.cat([x1,x2],dim=1)
        return self.block(x)

"""
    Conv block for U-net with InstanceNorm and LeakyReLU for downscaling
"""
class ConvBlockwINLR(nn.Module) :
    def __init__(self,in_ch,out_ch) :
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(out_ch,out_ch,kernel_size=3,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.InstanceNorm2d(out_ch)
        )
    
    def forward(self,x) :
        return self.block(x)

"""
    Conv block for U-net with InstanceNorm
"""
class ConvBlockwIN(nn.Module) :
    def __init__(self,in_ch,out_ch) :
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(out_ch)
        )
    
    def forward(self,x) :
        return self.block(x)

"""
    Downconv block for U-net with InstanceNorm and LeakyReLU for downscaling
"""
class DownConvBlockwIN(nn.Module) :
    def __init__(self,in_ch,out_ch) :
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,in_ch,kernel_size=3,stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            ConvBlockwINLR(in_ch,out_ch)
        )
        
    def forward(self,x) :
        return self.block(x)

"""
    Upconv block without concatenation for U-net
"""
class UpConvBlockNoConcatwIN(nn.Module) :
    def __init__(self,in_ch,out_ch) :
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_ch,in_ch),
            nn.ConvTranspose2d(in_ch,out_ch,kernel_size=2,stride=2),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(out_ch)
        )
        
    def forward(self,x) :
        return self.block(x)
    
"""
    Upconv block with concat for U-net
"""
class UpConvBlockwIN(nn.Module) :
    def __init__(self,in_ch,out_ch) :
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(2*in_ch,in_ch),
            nn.ConvTranspose2d(in_ch,out_ch,kernel_size=2,stride=2),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(out_ch)
        )
        
    def forward(self,x1,x2) :
        x = torch.cat([x1,x2],dim=1)
        return self.block(x)

"""
    ConvCatBlock for DBSnet using IN
"""
class ConvCatBlockIN(nn.Module) :
    def __init__(self,ch_num) :
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_num,ch_num,3,1,1,padding_mode='circular'),
            nn.InstanceNorm2d(ch_num),
            nn.ReLU(True),
            nn.Conv2d(ch_num,ch_num,3,1,1,padding_mode='circular'),
            nn.InstanceNorm2d(ch_num)
        )
    
    def forward(self,x) :
        return torch.cat([x,self.block(x)],dim=1)

class CINormDSBlock(nn.Module) :
    def __init__(self,in_ch,out_ch) :
        super().__init__()

        self.conv = nn.Conv2d(in_ch,out_ch,3,2,1,padding_mode='circular')
        self.relu = nn.ReLU(True)

    def forward(self,x,gamma,beta) :
        x1 = self.conv(x)
        x2 = F.instance_norm(x1)

        g = gamma.view(x.size(0),-1,1,1)
        b = beta.view(x.size(0),-1,1,1)

        return self.relu(x2*g+b)

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

        x3 = self.conv2(self.relu(x2*g1+b1))

        g2 = gamma2.view(x.size(0),-1,1,1)
        b2 = beta2.view(x.size(0),-1,1,1)

        return torch.cat([x,x3*g2+b2],dim=1)