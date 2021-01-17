from torch import nn
import torch

"""
    Conv block
"""
class ConvBlock(nn.Module) :
    def __init__(self,in_ch,out_ch) :
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3,padding=1,padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,kernel_size=3,padding=1,padding_mode='circular'),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x) :
        return self.block(x)

"""
    Conv block w LeakyReLU
"""
class ConvBlockLR(nn.Module) :
    def __init__(self,in_ch,out_ch) :
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3,padding=1,padding_mode='circular'),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(out_ch,out_ch,kernel_size=3,padding=1,padding_mode='circular'),
            nn.LeakyReLU(0.2,inplace=True)
        )
    
    def forward(self,x) :
        return self.block(x)

"""
    Output conv block
"""
class OutBlock(nn.Module) :
    def __init__(self,in_ch,out_ch) :
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_ch,in_ch),
            nn.Conv2d(in_ch,out_ch,kernel_size=1)
        )
    
    def forward(self,x) :
        return self.block(x)

"""
    Conv block with InstanceNorm
"""
class ConvBlockwIN(nn.Module) :
    def __init__(self,in_ch,out_ch) :
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3,padding=1,padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,kernel_size=3,padding=1,padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(out_ch)
        )
    
    def forward(self,x) :
        return self.block(x)

"""
    Conv block with InstanceNorm and LeakyReLU for downscaling
"""
class ConvBlockwINLR(nn.Module) :
    def __init__(self,in_ch,out_ch) :
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3,padding=1,padding_mode='circular'),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(out_ch,out_ch,kernel_size=3,padding=1,padding_mode='circular'),
            nn.LeakyReLU(0.2,inplace=True),
            nn.InstanceNorm2d(out_ch)
        )
    
    def forward(self,x) :
        return self.block(x)

"""
    Downconv block for with InstanceNorm and LeakyReLU for downscaling
"""
class DownConvBlockwIN(nn.Module) :
    def __init__(self,in_ch,out_ch) :
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,in_ch,kernel_size=3,stride=2,padding=1,padding_mode='circular'),
            nn.LeakyReLU(0.2,inplace=True),
            ConvBlockwINLR(in_ch,out_ch)
        )
        
    def forward(self,x) :
        return self.block(x)

"""
    Upconv block without concatenation
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
    ResBlock for DBSnet based on Johnson et al.'s style transfer network
"""
class ResBlock(nn.Module) :
    def __init__(self,ch_num) :
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_num,ch_num,3,1,1,padding_mode='circular',bias=False),
            nn.BatchNorm2d(ch_num),
            nn.ReLU(True),
            nn.Conv2d(ch_num,ch_num,3,1,1,padding_mode='circular',bias=False),
            nn.BatchNorm2d(ch_num)
        )
    
    def forward(self,x) :
        return x+self.block(x)

"""
    Style transfer network inspired from Johnson et al
"""
class StyleNet(nn.Module) :
    def __init__(self,in_ch=1,out_nch=1,nf=16,dn_lv=2,n_res=5) :
        super().__init__()

        L = []
        self.in_ch = in_ch
        self.out_nch = out_nch
        out_ch = nf
        for i in range(dn_lv) :
            L += [nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=2,padding=1,\
                padding_mode='circular',bias=False),\
                nn.BatchNorm2d(out_ch),\
                nn.ReLU(True)]
            in_ch = out_ch
            out_ch = out_ch*2
        
        for i in range(n_res) :
            L += [ResBlock(in_ch)]
        
        out_ch = in_ch // 2
        for i in range(dn_lv) :
            L += [nn.ConvTranspose2d(in_ch,out_ch,kernel_size=2,stride=2,bias=False),\
                nn.BatchNorm2d(out_ch),\
                nn.ReLU(True)]
            in_ch = out_ch
            out_ch = out_ch // 2
        
        L += [nn.Conv2d(in_ch,self.out_nch,3,1,1,padding_mode='circular'),nn.ReLU(True)]

        self.block = nn.Sequential(*L)

    def forward(self,x) :
        return self.block(x)

"""
    ResBlock for DBSnet based on Johnson et al.'s style transfer network using IN
"""
class ResBlockIN(nn.Module) :
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
        return x+self.block(x)

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

"""
    Style transfer network inspired from Johnson et al
    Replaced BN to IN everywhere, added Conv in front, replaced ResBlock with ConvBlockIN
"""
class StyleNetIN(nn.Module) :
    def __init__(self,in_ch=1,out_nch=1,dn_lv=2,n_res=5) :
        super().__init__()

        self.in_ch = in_ch
        self.out_nch = out_nch
        out_ch = 16
        L = [nn.Conv2d(in_ch,out_ch,3,1,1,padding_mode='circular'),nn.ReLU(True)]
        in_ch = out_ch
        out_ch = out_ch*2
        for i in range(dn_lv) :
            L += [nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=2,padding=1,\
                padding_mode='circular'),\
                nn.InstanceNorm2d(out_ch),\
                nn.ReLU(True)]
            in_ch = out_ch
            out_ch = out_ch*2
        
        for i in range(n_res) :
            L += [ConvBlockwIN(in_ch,in_ch)]
        
        out_ch = in_ch // 2
        for i in range(dn_lv) :
            L += [nn.ConvTranspose2d(in_ch,out_ch,kernel_size=2,stride=2),\
                nn.InstanceNorm2d(out_ch),\
                nn.ReLU(True)]
            in_ch = out_ch
            out_ch = out_ch // 2
        
        L += [nn.Conv2d(in_ch,self.out_nch,3,1,1,padding_mode='circular'),nn.ReLU(True)]

        self.block = nn.Sequential(*L)

    def forward(self,x) :
        return self.block(x)

"""
    Style transfer network inspired from Johnson et al
    Replaced BN to IN everywhere, added Conv in front, replaced ResBlock with ConvCatBlockIN
"""
class StyleNetINCat(nn.Module) :
    def __init__(self,in_ch=1,out_nch=1,nf=16,dn_lv=1,n_res=3) :
        super().__init__()

        self.in_ch = in_ch
        self.out_nch = out_nch
        out_ch = nf
        L = [nn.Conv2d(in_ch,out_ch,3,1,1,padding_mode='circular'),nn.ReLU(True)]
        in_ch = out_ch
        out_ch = out_ch*2
        for i in range(dn_lv) :
            L += [nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=2,padding=1,\
                padding_mode='circular'),\
                nn.InstanceNorm2d(out_ch),\
                nn.ReLU(True)]
            in_ch = out_ch
            out_ch = out_ch*2
        
        for i in range(n_res) :
            L += [ConvCatBlockIN(in_ch)]
            in_ch = in_ch*2
        
        out_ch = in_ch // 2
        for i in range(dn_lv) :
            L += [nn.ConvTranspose2d(in_ch,out_ch,kernel_size=2,stride=2),\
                nn.InstanceNorm2d(out_ch),\
                nn.ReLU(True)]
            in_ch = out_ch
            out_ch = out_ch // 2
        
        L += [nn.Conv2d(in_ch,self.out_nch,3,1,1,padding_mode='circular'),nn.ReLU(True)]

        self.block = nn.Sequential(*L)

    def forward(self,x) :
        return self.block(x)

"""
    Encoder for AE-GAN based implementation
"""
class EncoderWB(nn.Module) :
    def __init__(self,in_ch=3,out_nc=16,input_size=256) :
        super().__init__()
        self.nc = out_nc
        self.block = nn.Sequential(
            ConvBlockLR(in_ch,12),
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
            ConvBlockLR(in_ch,12),
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
    Encoder for AE-GAN based implementation
"""
class Encoder2(nn.Module) :
    def __init__(self,in_ch=3,out_nc=16,input_size=256) :
        super().__init__()
        self.nc = out_nc
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,12,3,1,1,padding_mode='circular'),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(12,12,3,1,1,padding_mode='circular'),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(12,24,3,2,1,padding_mode='circular'),
            nn.LeakyReLU(0.2,True),
            nn.InstanceNorm2d(24),
            nn.Conv2d(24,48,3,2,1,padding_mode='circular'),
            nn.LeakyReLU(0.2,True),
            nn.InstanceNorm2d(48),
            nn.Conv2d(48,96,3,2,1,padding_mode='circular'),
            nn.LeakyReLU(0.2,True),
            nn.InstanceNorm2d(96),
            nn.Conv2d(96,192,3,2,1,padding_mode='circular'),
            nn.LeakyReLU(0.2,True),
            nn.InstanceNorm2d(192),
            nn.Conv2d(192,64,1,1)
        )

        self.fc = nn.Linear(((input_size//(2**4))**2)*64,out_nc)
    
    def forward(self,x) :
        xc = self.block(x)
        xv = xc.view(xc.size(0),-1)
        out = self.fc(xv)
        return out

class EncoderVAE(nn.Module) :
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

        self.fc1 = nn.Linear(((input_size//(2**4))**2)*16,out_nc)
        self.fc2 = nn.Linear(((input_size//(2**4))**2)*16,out_nc)
    
    def forward(self,x) :
        xc = self.block(x)
        xv = xc.view(xc.size(0),-1)
        out1 = self.fc1(xv)
        out2 = self.fc2(xv)
        return out1, out2

class DecoderWB(nn.Module) :
    def __init__(self,in_nc=16,out_ch=3,output_size=256) :
        super().__init__()
        self.nc = in_nc
        self.inner_size = output_size//(2**4)
        self.fc = nn.Linear(in_nc,(self.inner_size**2)*192)
        self.block = nn.Sequential(
            UpConvBlockNoConcatwIN(192,96),
            UpConvBlockNoConcatwIN(96,48),
            UpConvBlockNoConcatwIN(48,24),
            UpConvBlockNoConcatwIN(24,12),
            OutBlock(12,out_ch)
        )
    
    def forward(self,x) :
        xc = self.fc(x)
        xv = xc.view(xc.size(0),192,self.inner_size,self.inner_size)
        out = self.block(xv)
        return out

class Decoder2(nn.Module) :
    def __init__(self,in_nc=16,out_ch=3,output_size=256) :
        super().__init__()
        self.nc = in_nc
        self.inner_size = output_size//(2**4)
        self.fc = nn.Linear(in_nc,(self.inner_size**2)*64)
        self.block = nn.Sequential(
            nn.Conv2d(64,192,1,1),
            nn.ReLU(True),
            nn.ConvTranspose2d(192,96,2,2),
            nn.ReLU(True),
            nn.InstanceNorm2d(96),
            nn.ConvTranspose2d(96,48,2,2),
            nn.ReLU(True),
            nn.InstanceNorm2d(48),
            nn.ConvTranspose2d(48,24,2,2),
            nn.ReLU(True),
            nn.InstanceNorm2d(24),
            nn.ConvTranspose2d(24,12,2,2),
            nn.ReLU(True),
            nn.InstanceNorm2d(12),
            nn.Conv2d(12,12,3,1,1,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(12,1,1,1)
        )
    
    def forward(self,x) :
        xc = self.fc(x)
        xv = xc.view(xc.size(0),64,self.inner_size,self.inner_size)
        out = self.block(xv)
        return out

class DiscAAE(nn.Module) :
    def __init__(self,in_dim=16) :
        super().__init__()
        self.nc = in_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim,16),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(16,1)
        )
    
    def forward(self,x) :
        return self.fc(x)

"""
    Discriminator for GAN-based implementation
    Based on the PatchGAN structure in https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    Added Dropout for slower discriminator fitting
"""
class Discriminator(nn.Module) :
    def __init__(self,in_ch=3,ndf=64) :
        super().__init__()
        
        kw = 4
        padw = 1
        
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,ndf,kernel_size=kw,stride=2,padding=padw,padding_mode='circular'),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf,ndf*2,kernel_size=kw,stride=2,padding=padw,bias=False,padding_mode='circular'),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf*2,ndf*4,kernel_size=kw,stride=2,padding=padw,bias=False,padding_mode='circular'),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf*4,ndf*8,kernel_size=kw,stride=2,padding=padw,bias=False,padding_mode='circular'),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf*8,1,kernel_size=kw,stride=1,padding=padw,padding_mode='circular')
        )
    
    def forward(self,x) :
        return self.block(x)

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
    
    def forward(self,x) :
        return self.block(x)

"""
    Pixel-level Discriminator for GAN-based implementation
    Added Dropout for slower discriminator fitting
"""
class PixDisc(nn.Module) :
    def __init__(self,in_ch=1,ndf=64) :
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,ndf,kernel_size=1,stride=1),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf,ndf,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf,1,kernel_size=1,stride=1)
        )
    
    def forward(self,x) :
        return self.block(x)

class DenseStyleNet(nn.Module) :
    def __init__(self,in_ch=1,out_ch=1,ndf=16) :
        super().__init__()

        self.inblock = ConvBlock(in_ch,ndf)
        self.style1 = StyleNet(ndf,ndf,dn_lv=1,n_res=2)
        self.style2 = StyleNet(2*ndf,ndf,dn_lv=1,n_res=2)
        self.outblock = OutBlock(3*ndf,out_ch)
    
    def forward(self,x) :
        x1 = self.inblock(x)
        x2 = self.style1(x1)
        x2c = torch.cat([x1,x2],dim=1)
        x3 = self.style2(x2c)
        x3c = torch.cat([x1,x2,x3],dim=1)
        out = self.outblock(x3c)

        return out
    
"""
    Pyramid Discriminator class
    For each level feature extractors are shared before last convolution
"""
class PyramidDisc(nn.Module) :
    def __init__(self,in_ch=1,out_ch=1) :
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch,ndf,kernel_size=1,stride=1),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1,bias=False,padding_mode='circular'),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,True),
            nn.Dropout()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1,bias=False,padding_mode='circular'),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2,True),
            nn.Dropout()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf*4,ndf*8,kernel_size=4,stride=2,padding=1,bias=False,padding_mode='circular'),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2,True),
            nn.Dropout()
        )
        self.out1 = nn.Conv2d(ndf*2,1,kernel_size=1,stride=1)
        self.out2 = nn.Conv2d(ndf*4,1,kernel_size=1,stride=1)
        self.out3 = nn.Conv2d(ndf*8,1,kernel_size=1,stride=1)
    
    def forward(self,x) :
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        return self.out1(x1),self.out2(x2),self.out3(x3)
        


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




