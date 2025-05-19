import torch 
import torch.nn as nn
from torch.utils.data import DataLoader , TensorDataset 
import wandb 
import numpy as np 
import os 
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
#Loading the data 
def load_mnist_local(data_dir):
    def read_idx3_ubyte(file_path): #Images 
        with open(file_path, 'rb') as f:
            magic_number = int.from_bytes(f.read(4), 'big')
            num_images = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
        return data
    def read_idx1_ubyte(file_path): #Labels 
        with open(file_path, 'rb') as f:
            magic_number = int.from_bytes(f.read(4), 'big')
            num_items = int.from_bytes(f.read(4), 'big')
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data
    train_images = read_idx3_ubyte(os.path.join(data_dir, 'train-images.idx3-ubyte'))
    train_labels = read_idx1_ubyte(os.path.join(data_dir, 'train-labels.idx1-ubyte'))
    test_images = read_idx3_ubyte(os.path.join(data_dir, 't10k-images.idx3-ubyte'))
    test_labels = read_idx1_ubyte(os.path.join(data_dir, 't10k-labels.idx1-ubyte'))
    return train_images, train_labels, test_images, test_labels

data_dir = "your data directory "
train_images , train_labels , test_images , test_labels = load_mnist_local(data_dir)

# Convert to PyTorch tensors 
train_images  = torch.tensor(train_images , dtype = torch.float32).unsqueeze(1) / 255.0 # (B , 1 , 28 , 28) 
test_images = torch.tensor(test_images , dtype = torch.float32).unsqueeze(1) / 255.0 
train_labels = torch.tensor(train_labels , dtype = torch.long)
test_labels = torch.tensor(test_labels , dtype = torch.long) 

# Create Dataloader 
train_dataset = TensorDataset(train_images , train_labels)
test_dataset = TensorDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset , batch_size = 64 , shuffle = True)
test_loader = DataLoader(test_dataset , batch_size = 64 , shuffle = False)

# Initialize Models 
teacher_model = TeacherModel()
student_model = StudentModel()
#Optimizers
teacher_optimizer = torch.optim.RMSprop(teacher_model.parameters() , lr = 0.001)
student_optimizer = torch.optim.Adam(student_model.parameters() , lr = 0.01)
# Loss Function 
def distillation_loss(student_output , teacher_output , temperature ): 
    #Cross Entropy Loss 
    pass