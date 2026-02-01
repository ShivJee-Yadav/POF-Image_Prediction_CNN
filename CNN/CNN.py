# Dataset is taken by camera having 7.5hz frame rate and 
# each image is captured at 8hz
# using formula to approximiate image = F = 1  + [i*(830/1000)] 
# total image in 1000s = 831 images 

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms , models
from PIL import Image
import pandas as pd
import os

# Dataset Load
class FrequencyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, discrete_multiplier=5):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # oversample discrete images
        discrete = self.df[self.df["source"] == "discrete"]
        sweep = self.df[self.df["source"] == "sweep"]

        self.df = pd.concat(
            [sweep, pd.concat([discrete]*discrete_multiplier, ignore_index=True)],
            ignore_index=True
        ).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_name).convert("L")

        # scale labels
        freq = row["frequency_hz"] / 22000.0
        freq = torch.tensor([freq], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, freq
    

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class RegressionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 1)
        self.act = nn.Sigmoid()  # constrain to [0,1]

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        return x

# Load Data
train_ds = FrequencyDataset(
    "CNN/all_labels.csv",
    "Final_Sweep_Reading",
    transform=transform,
    discrete_multiplier=4
)

print("Dataset length:", len(train_ds))

train_loader = DataLoader(train_ds , batch_size=16 , shuffle=True)

# CNN Model 
model = models.resnet18(weights="IMAGENET1K_V1")

w = model.conv1.weight.sum(dim=1, keepdim=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
model.conv1.weight = nn.Parameter(w)
model.fc = RegressionHead()

model = model.cuda()
print("Model device:", next(model.parameters()).device)

# Training Setup
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# -----------------------------
# EARLY STOPPING SETTINGS
# -----------------------------
patience = 10          # stop if no improvement for 10 epochs
best_loss = float("inf")
patience_counter = 0

print("Training Start")

# training Loop 
for epoch in range(200):   # allow many epochs, early stopping will stop automatically
    model.train()
    running_loss = 0.0

    for imgs, freqs in train_loader:
        imgs = imgs.cuda()
        freqs = freqs.cuda()

        preds = model(imgs)
        loss = criterion(preds, freqs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss = {epoch_loss:.6f}")

    # -----------------------------
    # EARLY STOPPING CHECK
    # -----------------------------
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        patience_counter = 0
        torch.save(model.state_dict(), "frequency_regression_cnn.pth")
        print("  → Improved, model saved.")
    else:
        patience_counter += 1
        print(f"  → No improvement ({patience_counter}/{patience})")

    if patience_counter >= patience:
        print("\nEarly stopping triggered — training stopped.")
        break