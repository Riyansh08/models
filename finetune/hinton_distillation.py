import torch 
import torch.nn as nn
from torch.utils.data import DataLoader , TensorDataset 

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
    # Get the logits 
    # KL Divergence
    soft_student = torch.softmax(student_output / temperature , dim = -1) 
    soft_teacher = torch.softmax(teacher_output / temperature , dim  = -1)
    # KL Divergence 
    kl = torch.sum(soft_teacher * (torch.log(soft_teacher + 1e-7) - torch.log(soft_student + 1e-7))) # Adding epsilon to avoid log(0)
    return torch.mean(kl) * temperature ** 2 # Normalize by temperature squared 
def cross_entropy(student_output , target):
    return F.cross_entropy(student_output , target)
#Total Loss 
def total_loss(student_output , teacher_output , target , temperature = 2.0  , alpha = 0.4): # Higher alpha means more emphaiss on distillation  # Higher temperature means more softened predictions , kind of like a regularization 
    loss_soft = distillation_loss(student_output , teacher_output , temperature )
    loss_hard = cross_entropy(student_output , target )
    return alpha * loss_hard + (1 - alpha) * loss_soft

#Training LOOPs 
def train_teacher():
    for i , (images, labels) in enumerate(train_loader):
        target = torch.nn.functional.one_hot(labels , num_classes = 10)
        # Setting optimizer to zero 
        teacher_optimizer.zero_grad()
        #Forward pass 
        teacher_output = teacher_model(images)
        loss_teacher = F.cross_entropy(teacher_output , target)
        #Backward pass
        loss_teacher.backward()
        teacher_optimizer.step()
        if i % 10 == 0:
            _ ,preds = torch.max(teacher_output , 1)
            correct = (preds == labels).sum().item()
            accuracy = correct / images.size(0)
            print(f"Epoch {i} , Step {i} , Loss {loss_teacher.item()} , Accuracy {accuracy}")
def train_student():
    for i , (images , labels) in enumerate(train_loader):
        target = torch.nn.functional.one_hot(labels , num_classes = 10)
        student_optimizer.zero_grad()
        student_output , _ = student_model(images)
        with torch.no_grad():
            teacher_output = teacher_model(images)
    
        loss_student = total_loss(student_output , teacher_output, target)
        loss_student.backward()
        student_optimizer.step()
        if i % 10 == 0:
            _ , preds = torch.max(student_output , 1)
            correct = (preds == labels).sum().item()
            accuracy = correct / images.size(0)
            print(f"Epoch {i} , Step {i} , Loss {loss_student.item()} , Accuracy {accuracy}")
            
# Test LOOPs
# @torch.no_grad()
def test_model(model , test_loader , is_teacher = False):
    model.eval()
    correct = 0 
    total = 0
    with torch.no_grad():
        for images , labels in test_loader:
            if is_teacher:
                output = model(images)
            else:
                output  , _ = model(images)
            _ , preds = torch.max(output , 1)
            total+=labels.size(0)
            correct+=(preds == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy of the model is {accuracy}%")
    return accuracy
   
print("Teacher Training Started...")
train_teacher()
print("Teacher Training Ended...")
print("Student Training Started...")
train_student()
print("Student Training Ended...")

teacher_accuracy = test_model(teacher_model, test_loader, is_teacher=True)
student_accuracy = test_model(student_model, test_loader, is_teacher=False)
     