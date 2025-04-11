#VAE - using U-Net architecture
#NOTE - dataset loading and training loop not included
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import init as init
import numpy as np 
from dataclasses import dataclass, field , asdict
from typing import List, Optional, Tuple
# NOTE - output size - [input - Kernel + 2*padding] / stride + 1
@dataclass  # Configuration class for the VAE model - can be adjusted for different datasets
class Config:
    """
    Configuration class for the VAE model.
    """
    batch_size : int = 64
    num_epochs : int 
    learning_rate : float = 0.01 
    x_dim : int 
    z_dim : int
    hidden_dim1 : int
    hidden_dim2 : int
    hidden_dim3 : int
    dropout : float = 0.2
    in_channels : int
    out_channelss : int
    kernel_size :int
    stride : int
    padding : int 
    
#LeakyReLU (Custom Implementation)
class LeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2, inplace=False):
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace
    def forward(self, x):
        return F.leaky_relu(x, negative_slope=self.negative_slope, inplace=self.inplace)  
class Encoder(nn.Module):
    def __init__(self, config:Config):
        super(Encoder,self).__init__()
        self.conv1=nn.Conv2d(config.in_channels,config.hidden_dim1,kernel_size=config.kernel_size,stride=config.stride,padding=config.padding)
        self.conv2=nn.Conv2d(config.hidden_dim1,config.hidden_dim2,kernel_size=config.kernel_size,stride=config.stride,padding=config.padding)
        self.conv3=nn.Conv2d(config.hidden_dim2,config.hidden_dim3,kernel_size=config.kernel_size,stride=config.stride,padding=config.padding)
        self.flatten=nn.Flatten()
        dummy=torch.zeros(1,config.in_channels,config.x_dim,config.x_dim)
        with torch.no_grad():
            dummy_out=self.forward_conv(dummy)
            self.flattened_size=dummy_out.view(1,-1).shape[1]
        self.fc_mu=nn.Linear(self.flattened_size,config.z_dim)
        self.fc_logvar=nn.Linear(self.flattened_size,config.z_dim)
        self.leaky_relu=nn.LeakyReLU(0.2)
    def forward_conv(self,x):
        x=self.leaky_relu(self.conv1(x))
        x=self.leaky_relu(self.conv2(x))
        x=self.leaky_relu(self.conv3(x))
        return x
    def forward(self,x):
        x=self.forward_conv(x)
        x=self.flatten(x)
        mu=self.fc_mu(x)
        logvar=self.fc_logvar(x)
        return mu,logvar
#DECODER -- using U-Net architecture
class Decoder(nn.Module):
    def __init__(self,config:Config):
        super(Decoder,self).__init__()
        self.config=config
        self.fc=nn.Linear(config.z_dim,config.hidden_dim3*4*4)
        self.deconv1=nn.ConvTranspose2d(config.hidden_dim3,config.hidden_dim2,kernel_size=config.kernel_size,stride=config.stride,padding=config.padding,output_padding=1)
        self.deconv2=nn.ConvTranspose2d(config.hidden_dim2,config.hidden_dim1,kernel_size=config.kernel_size,stride=config.stride,padding=config.padding,output_padding=1)
        self.deconv3=nn.ConvTranspose2d(config.hidden_dim1,config.out_channelss,kernel_size=config.kernel_size,stride=config.stride,padding=config.padding,output_padding=1)
        self.leaky_relu=nn.LeakyReLU(0.2)
        self.sigmoid=nn.Sigmoid()
    def forward(self,z):
        x=self.fc(z)
        x=x.view(-1,self.config.hidden_dim3,4,4)
        x=self.leaky_relu(self.deconv1(x))
        x=self.leaky_relu(self.deconv2(x))
        x=self.sigmoid(self.deconv3(x))
        return x
#VAE - Variational Autoencoder
class VAE(nn.Module):
    def __init__(self,config:Config):
        super(VAE,self).__init__()
        self.encoder=Encoder(config)
        self.decoder=Decoder(config)
    def reparameterize(self,mu,logvar):
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        return mu+eps*std
    def forward(self,x):
        mu,logvar=self.encoder(x)
        z=self.reparameterize(mu,logvar)
        recon=self.decoder(z)
        return recon,mu,logvar
#Loss Function
def loss_function(recon_x,x,mu,logvar):
    BCE=F.binary_cross_entropy(recon_x,x,reduction='sum')
    KLD=-0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    return BCE+KLD
