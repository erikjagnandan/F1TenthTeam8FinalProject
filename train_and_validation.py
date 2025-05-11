import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Dataset class
class ImitationDrivingImageDataset(Dataset):
    def __init__(self, csv_path, image_transform=None):
        self.data = pd.read_csv(csv_path)
        self.image_transform = image_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]['image_path']
        if image_path[0] == '~':
            image_path = '/home/ejag/' + image_path[2:]
        image = Image.open(image_path).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)
        steering = float(self.data.iloc[idx]['steering_angle'])
        speed = float(self.data.iloc[idx]['speed'])
        target = torch.tensor([steering, speed], dtype=torch.float32)
        return {"input": image, "target": target}

# Model class
class NeRFImitationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet18(pretrained=False)
        self.encoder.fc = nn.Sequential(
            nn.Linear(self.encoder.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.encoder(x)
        steering = out[:, 0] * (np.pi / 4)
        speed = (out[:, 1] + 1) * 2
        return steering, speed

# Transforms
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load dataset and split
dataset = ImitationDrivingImageDataset("dataset.csv", image_transform=image_transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model, loss, optimizer setup
num_epochs = 40
lam = 10.0
learning_rates = {0: 1e-4, 10: 3e-5, 20: 1e-5, 30: 3e-6}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeRFImitationNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rates[0])

train_losses = []
val_losses = []

# Training and validation loop
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        images = batch["input"].to(device)
        targets = batch["target"].to(device)
        steering_true = targets[:, 0]
        speed_true = targets[:, 1]

        steering_pred, speed_pred = model(images)
        loss_steering = criterion(steering_pred, steering_true)
        loss_speed = criterion(speed_pred, speed_true)
        loss = loss_steering + lam * loss_speed

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * images.size(0)
    avg_train_loss = total_train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            images = batch["input"].to(device)
            targets = batch["target"].to(device)
            steering_true = targets[:, 0]
            speed_true = targets[:, 1]

            steering_pred, speed_pred = model(images)
            loss_steering = criterion(steering_pred, steering_true)
            loss_speed = criterion(speed_pred, speed_true)
            loss = loss_steering + lam * loss_speed

            total_val_loss += loss.item() * images.size(0)
    avg_val_loss = total_val_loss / len(val_loader.dataset)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    if epoch in learning_rates:
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rates[epoch]

# Save the model
torch.save(model.state_dict(), "nerf_imitation_model_lam_1.pth")

# Plotting the losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs+1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png")
plt.show()
