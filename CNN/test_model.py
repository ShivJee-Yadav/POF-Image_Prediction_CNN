import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# SAME Regression Head as training
# -----------------------------
class RegressionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        return x

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
        pred = model(image)

    # un-normalize: convert [0,1] → [0,1000]
    freq_hz = pred.item() * 22000.0
    return freq_hz

# -----------------------------
# Test on one image
# -----------------------------
stri = ["256_hertz/temp-12012025172707-2.Bmp",
        "32_hertz/temp-12012025172548-1.Bmp", 
        "256_hertz/temp-12012025172706-0.Bmp",
         "512_hertz/temp-12012025172729-1.Bmp",
         "22000_hertz/temp-12012025174411-7.Bmp",
         "22000_hertz/temp-12012025174410-4.Bmp",
         "16384_hertz/temp-12012025174324-3.Bmp"]
print("\n")
for i in range(len(stri)):

    result = predict(stri[i])

    if(result < 250):
        print("Original Freq: ",stri[i].rsplit("/", 1)[0])
        print("Predicted Range: Low Frequency (range : 1hz - 250hz)")
    elif (result >= 250 and result < 2000):
        print("Original Freq: ",stri[i].rsplit("/", 1)[0])
        print("Predicted Range: Middle Frequencies (range : 2kz to 20khz )")
    elif (result >=2000 and result < 20000):
        print("Original Freq: ",stri[i].rsplit("/", 1)[0])
        print("Predicted Range: HighFrequency (range : above 20khz) ")
    else:
        print("Original Freq: ",stri[i].rsplit("/", 1)[0])
        print("Predicted Range: Ultrasound Frequency")
    
    print("Predicted frequency:", result , "\n")