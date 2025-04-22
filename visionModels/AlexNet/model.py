import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS    = 90
BATCH_SIZE    = 128
LR_INIT       = 0.01
IMAGE_DIM     = 227
NUM_CLASSES   = 1000
DEVICE_IDS    = [0,1,2,3]

INPUT_ROOT_DIR  = 'alexnet_data_in'
TRAIN_IMG_DIR   = os.path.join(INPUT_ROOT_DIR, 'imagenet')
OUTPUT_DIR      = 'alexnet_data_out'
LOG_DIR         = os.path.join(OUTPUT_DIR, 'tblogs')
CHECKPOINT_DIR  = os.path.join(OUTPUT_DIR, 'models')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class AlexNet(nn.Module):
    """
    AlexNet as described in:
    'ImageNet Classification with Deep Convolutional Neural Networks'
    by Krizhevsky et al. (2012)
    """
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize conv layers
        for m in self.features:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)
        # As in the paper, set biases of conv2, conv4, conv5 to 1
        nn.init.constant_(self.features[4].bias, 1.0)
        nn.init.constant_(self.features[10].bias, 1.0)
        nn.init.constant_(self.features[12].bias, 1.0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        return self.classifier(x)


if __name__ == '__main__':
    # Reproducibility
    torch.manual_seed(42)
    seed = torch.initial_seed()
    print(f'Using seed: {seed}')

    # TensorBoard
    tbwriter = SummaryWriter(log_dir=LOG_DIR)
    print("TensorBoardX writer created")

    # Model
    model = AlexNet(num_classes=NUM_CLASSES).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=DEVICE_IDS)
    print("Model ready")

    # Data
    transform = transforms.Compose([
        transforms.CenterCrop(IMAGE_DIM),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])
    dataset = datasets.ImageFolder(TRAIN_IMG_DIR, transform=transform)
    dataloader = data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    print("DataLoader ready")

    # Optimizer & Scheduler
    optimizer    = optim.Adam(model.parameters(), lr=LR_INIT)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print("Optimizer and LR scheduler ready")

    # Training loop
    print("Starting training...")
    total_steps = 0
    for epoch in range(NUM_EPOCHS):
        lr_scheduler.step()
        for imgs, labels in dataloader:
            imgs   = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss    = F.cross_entropy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_steps += 1

            # Log every 10 steps
            if total_steps % 10 == 0:
                with torch.no_grad():
                    preds    = outputs.argmax(dim=1)
                    acc      = (preds == labels).float().mean().item()
                    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
                          f"Step [{total_steps}] "
                          f"Loss: {loss.item():.4f} "
                          f"Acc: {acc:.4f}")
                    tbwriter.add_scalar('train/loss', loss.item(), total_steps)
                    tbwriter.add_scalar('train/accuracy', acc, total_steps)

            # Histogram every 100 steps
            if total_steps % 100 == 0:
                with torch.no_grad():
                    print('-' * 30)
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            tbwriter.add_histogram(f'grads/{name}', param.grad.cpu().numpy(), total_steps)
                        tbwriter.add_histogram(f'weights/{name}', param.data.cpu().numpy(), total_steps)

        # Save checkpoint at end of each epoch
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'alexnet_epoch{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'total_steps': total_steps,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'seed': seed
        }, ckpt_path)
        print(f"Checkpoint saved at {ckpt_path}")

    tbwriter.close()
