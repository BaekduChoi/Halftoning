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
    U-net inspired by "Deep White-Balance Editing" by Afifi et al. (CVPR 2020)
    https://github.com/mahmoudnafifi/Deep_White_Balance
    Overall structure is similar but slightly different
"""
class Unet(nn.Module) :
    def __init__(self,in_ch=1,out_ch=2) :
        super().__init__()
        
        self.inBlock = ConvBlock(in_ch,24)
        self.downBlock1 = DownConvBlock(24,48)
        self.downBlock2 = DownConvBlock(48,96)
        self.downBlock3 = DownConvBlock(96,192)
        self.downBlock4 = DownConvBlock(192,384)
        
        self.upBlock1 = UpConvBlockNoConcat(384,192)
        self.upBlock2 = UpConvBlock(192,96)
        self.upBlock3 = UpConvBlock(96,48)
        self.upBlock4 = UpConvBlock(48,24)
        self.outBlock = OutBlock(24,out_ch)
        
    def forward(self,x) :
        x1 = self.inBlock(x)
        x2 = self.downBlock1(x1)
        x3 = self.downBlock2(x2)
        x4 = self.downBlock3(x3)
        x5 = self.downBlock4(x4)
        
        out = self.upBlock1(x5)
        out = self.upBlock2(out,x4)
        out = self.upBlock3(out,x3)
        out = self.upBlock4(out,x2)
        out = self.outBlock(out,x1)
        
        return out

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
    U-net with InstanceNorm for downsampling & upsampling blocks
    LeakyReLU activation for downsampling blocks
    Did not use normalization in input block and output block following DCGAN paper
"""
class UnetwIN(nn.Module) :
    def __init__(self,in_ch=1,out_ch=2) :
        super().__init__()
        
        self.inBlock = ConvBlock(in_ch,24)
        self.downBlock1 = DownConvBlockwIN(24,48)
        self.downBlock2 = DownConvBlockwIN(48,96)
        self.downBlock3 = DownConvBlockwIN(96,192)
        self.downBlock4 = DownConvBlockwIN(192,384)
        
        self.upBlock1 = UpConvBlockNoConcatwIN(384,192)
        self.upBlock2 = UpConvBlockwIN(192,96)
        self.upBlock3 = UpConvBlockwIN(96,48)
        self.upBlock4 = UpConvBlockwIN(48,24)
        self.outBlock = OutBlock(24,out_ch)
        
    def forward(self,x) :
        x1 = self.inBlock(x)
        x2 = self.downBlock1(x1)
        x3 = self.downBlock2(x2)
        x4 = self.downBlock3(x3)
        x5 = self.downBlock4(x4)
        
        out = self.upBlock1(x5)
        out = self.upBlock2(out,x4)
        out = self.upBlock3(out,x3)
        out = self.upBlock4(out,x2)
        out = self.outBlock(out,x1)
        
        return out

"""
    Encoder for AE-GAN based implementation
"""
class Encoder(nn.Module) :
    def __init__(self,in_ch=3,out_nc=16,input_size=256) :
        super().__init__()
        self.nc = out_nc
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,12,1,1),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(12,24,3,2,1),
            nn.LeakyReLU(0.2,True),
            nn.InstanceNorm2d(24),
            nn.Conv2d(24,48,3,2,1),
            nn.LeakyReLU(0.2,True),
            nn.InstanceNorm2d(48),
            nn.Conv2d(48,96,3,2,1),
            nn.LeakyReLU(0.2,True),
            nn.InstanceNorm2d(96),
            nn.Conv2d(96,192,3,2,1),
            nn.LeakyReLU(0.2,True),
            nn.InstanceNorm2d(192),
            nn.Conv2d(192,16,1,1)
        )

        self.fc = nn.Linear(((input_size//(2**4))**2)*16,out_nc)
    
    def forward(self,x) :
        xc = self.block(x)
        xv = xc.view(xc.size(0),-1)
        out = self.fc(xv)
        return out

"""
    Encoder for AE-GAN based implementation
"""
class EncoderWB(nn.Module) :
    def __init__(self,in_ch=3,out_nc=16,input_size=256) :
        super().__init__()
        self.nc = out_nc
        self.block = nn.Sequential(
            ConvBlock(in_ch,12),
            DownConvBlockwIN(12,24),
            DownConvBlockwIN(24,48),
            DownConvBlockwIN(48,96),
            DownConvBlockwIN(96,192)
        )

        self.fc = nn.Linear(((input_size//(2**4))**2)*192,out_nc)
    
    def forward(self,x) :
        xc = self.block(x)
        xv = xc.view(xc.size(0),-1)
        out = self.fc(xv)
        return out

"""
    Encoder for AE-GAN based implementation, VAE-like
"""
class EncoderWBVAE(nn.Module) :
    def __init__(self,in_ch=3,out_nc=16,input_size=256) :
        super().__init__()
        self.nc = out_nc
        self.block = nn.Sequential(
            ConvBlock(in_ch,12),
            DownConvBlockwIN(12,24),
            DownConvBlockwIN(24,48),
            DownConvBlockwIN(48,96),
            DownConvBlockwIN(96,192)
        )

        self.fc1 = nn.Linear(((input_size//(2**4))**2)*192,out_nc)
        self.fc2 = nn.Linear(((input_size//(2**4))**2)*192,out_nc)
    
    def forward(self,x) :
        xc = self.block(x)
        xv = xc.view(xc.size(0),-1)
        out1 = self.fc1(xv)
        out2 = self.fc2(xv)
        return out1, out2

"""
    Discriminator for GAN-based implementation
    Based on the PatchGAN structure in https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    Added Dropout for slower discriminator fitting
"""
class Discriminator(nn.Module) :
    def __init__(self,input_ch=3,ndf=64) :
        super().__init__()
        
        kw = 4
        padw = 1
        
        self.block = nn.Sequential(
            nn.Conv2d(input_ch,ndf,kernel_size=kw,stride=2,padding=padw),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf,ndf*2,kernel_size=kw,stride=2,padding=padw,bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf*2,ndf*4,kernel_size=kw,stride=2,padding=padw,bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf*4,ndf*8,kernel_size=kw,stride=2,padding=padw,bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf*8,1,kernel_size=kw,stride=1,padding=padw)
        )
    
    def forward(self,x) :
        return self.block(x)

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
    Added noise for slower discriminator fitting
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
    
    def __call__(self,prediction,is_real,noise=0.1) :
        label = self.get_label(prediction,is_real)
        label.to(self.device)
        return self.loss(prediction+noise*torch.randn_like(prediction),label)

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

"""
    Style transfer network inspired from Johnson et al
    Replaced BN to CIN everywhere, added Conv in front, replaced ResBlock with ConvCatBlockCIN
"""
class StyleNetINCatCIN(nn.Module) :
    def __init__(self,in_ch=1,in2_ch=16,out_nch=1) :
        super().__init__()

        self.in_ch = in_ch
        self.out_nch = out_nch

        self.inblock = nn.Sequential(
            nn.Conv2d(in_ch,16,3,1,1,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(16,32,3,2,1,padding_mode='circular'),
            nn.InstanceNorm2d(32),
            nn.ReLU(True)
        )
        self.block1 = ConvCatBlockCIN(32)
        self.block2 = ConvCatBlockCIN(64)
        self.block3 = ConvCatBlockCIN(128)

        self.fc1 = nn.Sequential(
            nn.Linear(in2_ch,448)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in2_ch,448)
        )

        self.blockout = nn.Sequential(
            nn.ConvTranspose2d(256,64,kernel_size=2,stride=2),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64,self.out_nch,3,1,1,padding_mode='circular'),
            nn.ReLU(True)
        )

    def forward(self,x,z) :
        gammas = self.fc1(z)
        betas = self.fc2(z)
        x1 = self.inblock(x)
        x2 = self.block1(x1,gammas[:,:32],betas[:,:32],gammas[:,32:64],betas[:,32:64])
        x3 = self.block2(x2,gammas[:,64:128],betas[:,64:128],gammas[:,128:192],betas[:,128:192])
        x4 = self.block3(x3,gammas[:,192:320],betas[:,192:320],gammas[:,320:],betas[:,320:])

        return self.blockout(x4)