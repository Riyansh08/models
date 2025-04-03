import argparse
from model import ViTForClassfication
from utils.data import prepare_data 

exp_name = 'vit-with-10-epochs' 
batch_size = 32
epochs = 10 
lr = 1e-2 
save_model_every = 0  

import torch
from torch import nn, optim

device = "cuda" if torch.cuda.is_available() else "cpu"

config = {
    "patch_size": 4,  # Input a image size: 32x32 -> 8x8 patches
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10, # num_classes of CIFAR10
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
}

assert config["hidden_size"] % config["num_attention_heads"] == 0
assert config['intermediate_size'] == 4 * config['hidden_size']
assert config['image_size'] % config['patch_size'] == 0


class Trainer:
   
    def __init__(self, model, optimizer, loss_fn, exp_name, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device

    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
       
        # Keep track of the losses and accuracies
        train_losses, test_losses, accuracies = [], [], []
        # Train the model
        for i in range(epochs):
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            print(f"Epoch: {i+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
            if save_model_every_n_epochs > 0 and (i+1) % save_model_every_n_epochs == 0 and i+1 != epochs:
                print('\tSave checkpoint at epoch', i+1)
               

    def train_epoch(self, trainloader):
        self.model.train()
        total_loss = 0
        for batch in trainloader:
          
            batch = [t.to(self.device) for t in batch]
            images, labels = batch
            
            self.optimizer.zero_grad()
          
            loss = self.loss_fn(self.model(images)[0], labels)
           
            loss.backward()
            
            self.optimizer.step()
            total_loss += loss.item() * len(images)
        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in testloader:
               
                batch = [t.to(self.device) for t in batch]
                images, labels = batch

                
                logits, _ = self.model(images)

                
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)

             
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Utils")
   
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
    
    return parser.parse_args()
def main():
  
    args = parse_args()
    print(f"Starting experiment: {args.epochs} epochs")
    epochs = args.epochs
    save_model_every_n_epochs = save_model_every
   
    trainloader, testloader= prepare_data(batch_size=batch_size)
    
    model = ViTForClassfication(config)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, exp_name, device=device)
    trainer.train(trainloader, testloader, epochs, save_model_every_n_epochs=save_model_every_n_epochs)


if __name__ == '__main__':
    main()

