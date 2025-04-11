#VAE - using simple MLP 
#NOTE - dataset loading and training loop not included
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import init as init
import numpy as np
import math 
from torch.optim import AdamW
from dataclasses import dataclass
@dataclass
class Config:
    """
    Configuration class for the VAE model.
    """
    batch_size : int = 64
    num_epochs : int = 100
    learning_rate : float = 0.01
    x_dim :int = 784 # 28*28 mnist, can be altered for other datasets! 
    z_dim : int = 20
    hidden_dim1: int = 512
    hidden_dim2: int = 256
    dropout : float = 0.2
    
if torch.cuda.is_available():
     device = torch.device("cuda")
else:
    device = torch.device("cpu")   
class ENCODER(nn.Module):
    def __init__(self , config : Config):
        super(ENCODER , self).__init__()
        self.config = config
        
        self.fc1 = nn.Linear(config.x_dim , config.hidden_dim1)
        self.fc2 = nn.Linear(config.hidden_dim1 , config.hidden_dim2)
        self.fc_mean = nn.Linear(config.hidden_dim2 , config.z_dim)
        self.fc_logvar = nn.Linear(config.hidden_dim2 , config.z_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
    def forward(self , x):
        x = self.LeakyReLU(self.fc1(x))
        x = self.LeakyReLU(self.fc2(x))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)        
        return mean , logvar
class Decoder(nn.Module):
    def __init__(self , config : Config):
        super(Decoder , self).__init__()
        self.config = config 
        self.fc1 = nn.Linear(config.z_dim , config.hidden_dim2)
        self.fc2 = nn.Linear(config.hidden_dim2 , config.hidden_dim1)
        self.outputfc = nn.Linear(config.hidden_dim1 , config.x_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
    def forward(self, x ):
        x = self.LeakyReLU(self.fc1(x))
        x = self.LeakyReLU(self.fc2(x))
        x = self.outputfc(x)
        return x             
class VAE(nn.Module):
    def __init__(self , config : Config):
        super(VAE , self).__init__()
        self.config = config
        self.encoder = ENCODER(config)
        self.decoder = Decoder(config)
    def reparameterize(self , mean  , logvar):
        e = torch.randn_like(logvar)
        std = torch.exp(logvar * 0.5)
        return mean + std * e
        
    def forward(self ,x ):
        mean , logvar = self.encoder(x)
        e = self.reparameterize(mean , logvar)
        x_out = self.decoder(e)
        return x_out , mean , logvar       
#Loss Function
def loss_function(recon_x , x , mean , logvar):
    BCE = F.binary_cross_entropy(recon_x , x , reduction = 'sum')
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return BCE + KLD
