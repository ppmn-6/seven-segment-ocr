import os
import cv2
import torch
import torchmetrics
import torchvision
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.bitwise_not(image)
        label = torch.tensor(self.img_labels.iloc[idx, 1:8], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        #self.norm1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=32 * 14 * 14, out_features=num_classes)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

train_data = CustomImageDataset("/home/wang-zhisheng/Downloads/centad/dataset/training_new.csv",
                                   "/home/wang-zhisheng/Downloads/centad/dataset/training",
                                   transform=ToTensor())
test_data = CustomImageDataset("/home/wang-zhisheng/Downloads/centad/dataset/testing_new.csv",
                                   "/home/wang-zhisheng/Downloads/centad/dataset/testing",
                                   transform=ToTensor())

train_dataloader = DataLoader(train_data, batch_size=256, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=256, shuffle=False)

device = torch.device("cuda")
model = CNN(1, 7).to(device)
model.load_state_dict(torch.load("/home/wang-zhisheng/Downloads/centad/model3.pt"))
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-5)

print(device)
latest_loss = 1

num_epochs = 50
for epoch in range(num_epochs):
    print(f"Epoch [{epoch + 1}/100]")

    running_loss = 0.0
    total_samples = 0

    for batch_index, (data, targets) in enumerate(tqdm(train_dataloader)):
        data = data.to(device)
        targets = targets.to(device)
        scores = model(data)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        total_samples += data.size(0)
    
    #scheduler.step()
    epoch_loss = running_loss / total_samples
    print(f"Loss: {epoch_loss:.4f}")
    #print(f"LR: {scheduler.get_last_lr()[0]:.6f}")
    latest_loss = epoch_loss
    torch.save({
        "epoch": epoch + 1,
        "loss": latest_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        #"scheduler_state_dict": scheduler.state_dict(),
    }, f"/home/wang-zhisheng/Downloads/centad/model13_{epoch + 51}.pt")

'''device = torch.device("cuda")
model = CNN(1, 7)
model.load_state_dict(torch.load("/home/wang-zhisheng/Downloads/centad/model4.pt", weights_only=True)["model_state_dict"])
model.eval()

a = os.listdir("/home/wang-zhisheng/Downloads/centad/dataset/training")

with torch.no_grad():
   for file in os.listdir("/home/wang-zhisheng/Downloads/centad/dataset/training"):
    image = cv2.imread(f"/home/wang-zhisheng/Downloads/centad/dataset/training/{a[33829]}", cv2.IMREAD_GRAYSCALE)
    o = image
    image = torchvision.transforms.functional.to_tensor(image)
    image = image.unsqueeze(0)
    outputs = model(image)
    print(torch.sigmoid(outputs))
    plt.imshow(o)
    plt.show()
    break'''
