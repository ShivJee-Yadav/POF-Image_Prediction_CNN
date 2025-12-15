import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import math

# -----------------------------
# SAME Regression Head as training
# -----------------------------
class RegressionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 1)
        self.act = nn.Sigmoid()   # MUST match training

    def forward(self, x):
        return self.act(self.fc(x))


# -----------------------------
# Build model EXACTLY as in training
# -----------------------------
model = models.resnet18(weights="IMAGENET1K_V1")

# Convert conv1 to grayscale
w = model.conv1.weight.sum(dim=1, keepdim=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
model.conv1.weight = nn.Parameter(w)

# Attach SAME head
model.fc = RegressionHead()

# Load trained weights
state = torch.load("frequency_regression_cnn.pth")
model.load_state_dict(state)

model = model.cuda()
model.eval()

# -----------------------------
# SAME transforms as training
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# -----------------------------
# Prediction function
# -----------------------------
def predict(image_path):
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0).cuda()

    with torch.no_grad():
        pred = model(image).item()   # value in [0,1]

    # -----------------------------
    # INVERT LOG SCALE
    # -----------------------------
    max_log = math.log10(22000 + 1)

    # pred is normalized log → convert back to real Hz
    freq_hz = (10 ** (pred * max_log)) - 1

    return freq_hz


# -----------------------------
# Test on one image
# -----------------------------
test_image = "8192_hertz/temp-12012025174250-6.Bmp"
result = predict(test_image)
print("Predicted frequency:", result)