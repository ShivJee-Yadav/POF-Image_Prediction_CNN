import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
import math

# -----------------------------
# Frequency-dependent oversampling
# -----------------------------
def get_multiplier(freq):
    if freq < 300:
        return 10      # low frequencies are hardest
    elif freq < 2000:
        return 5       # mid frequencies
    else:
        return 3       # high frequencies
        

# -----------------------------
# Dataset
# -----------------------------
class FrequencyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Split sweep and discrete
        sweep = df[df["source"] == "sweep"]
        discrete = df[df["source"] == "discrete"]

        # Apply frequency-dependent oversampling
        expanded = []

        for _, row in discrete.iterrows():
            freq = row["frequency_hz"]
            m = get_multiplier(freq)
            for _ in range(m):
                expanded.append(row)

        # Combine sweep + oversampled discrete
        self.df = pd.concat([sweep, pd.DataFrame(expanded)], ignore_index=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_path).convert("L")

        # -----------------------------
        # LOG-SCALED LABEL
        # -----------------------------
        freq = row["frequency_hz"]
        freq_log = math.log10(freq + 1) / math.log10(22000 + 1)  # normalize to [0,1]
        freq_log = torch.tensor([freq_log], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, freq_log


# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# -----------------------------
# Regression Head
# -----------------------------
class RegressionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.fc(x))


# -----------------------------
# Load Data
# -----------------------------
train_ds = FrequencyDataset(
    "CNN/all_labels.csv",
    "Final_Sweep_Reading",
    transform=transform
)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
print("Dataset size:", len(train_ds))


# -----------------------------
# Model
# -----------------------------
model = models.resnet18(weights="IMAGENET1K_V1")

# grayscale conv1
w = model.conv1.weight.sum(dim=1, keepdim=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
model.conv1.weight = nn.Parameter(w)

model.fc = RegressionHead()
model = model.cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# -----------------------------
# Training Loop
# -----------------------------
best_loss = float("inf")
patience = 10
counter = 0

for epoch in range(200):
    model.train()
    running = 0.0

    for imgs, freqs in train_loader:
        imgs = imgs.cuda()
        freqs = freqs.cuda()

        preds = model(imgs)
        loss = criterion(preds, freqs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += loss.item()

    epoch_loss = running / len(train_loader)
    print(f"Epoch {epoch+1}, Loss = {epoch_loss:.6f}")

    # Early stopping
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        counter = 0
        torch.save(model.state_dict(), "frequency_regression_cnn_log.pth")
        print("  → Improved, model saved.")
    else:
        counter += 1
        print(f"  → No improvement ({counter}/{patience})")

    if counter >= patience:
        print("Early stopping triggered.")
        break