import torch 
import torch.nn as nn
from torch.utils.data import DataLoader , TensorDataset 
import wandb 
import torch.nn.functional as F
# Simple CNN Based Teacher Model 
# H out  - (H in - K + 2P)/ S  ) + 1
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel , self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1 , out_channels = 32 , kernel_size = 4 , stride = 1 , padding = 1)
        #maxpooling is not used , becuase we want to preserve the spatial information. (Maxpool = Loss of Information)
        # self.maxpool = nn.MaxPool2d(kernel_size = 2 , stride = 2) 
        # DROPOUT IS USED TO PREVENT OVERFITTING 
        self.dropout = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(in_channels = 32 , out_channels = 64 , kernel_size = 3 , stride = 1 , padding = 1)
        self.dropout2 = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(in_channels = 64 , out_channels = 128 , kernel_size = 3 , stride = 1 , padding = 1)
        self.dropout3 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128 * 3 *3  , 625)
        self.activation = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(625 , 10) # 10 CLASSES - 0-9
    
    def forward(self , x):
        x = self.activation(self.conv1(x))
        x = self.dropout(x)
        x = self.activation(self.conv2(x))
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.dropout3(x)
        x = x.view(x.size(0) , -1)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # size ( B , 10)
        return x 
  
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel , self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8 , kernel_size = 3 , stride = 1 , padding = 1)
        # 8 is the number of filters 
        # out - (28 - 3 + 2*1)/1 + 1 = 28
        self.fc1 = nn.Linear(8 * 28 * 28 , 512)
        self.fc2 = nn.Linear(512 , 256)
        self.activation = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(256 , 10)
    def forward(self , x):
        x = self.activation(self.conv1(x))
        x = x.view(x.size(0) , -1)
        x = self.activation(self.fc3(self.activation(self.fc2(self.activation(self.fc1(x))))))
        return x  , x.size # (B , 10)

    

    
        
        