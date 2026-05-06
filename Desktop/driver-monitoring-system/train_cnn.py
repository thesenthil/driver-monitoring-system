import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

print("Program started...")

# ================= DEBUG CHECK =================
train_path = r"C:\Users\senthil\Desktop\driver-monitoring-system\dataset\train"

print("Train path exists:", os.path.exists(train_path))

open_path = os.path.join(train_path, "open")
closed_path = os.path.join(train_path, "closed")

if os.path.exists(open_path):
    print("Open folder files:", os.listdir(open_path))
else:
    print("Open folder NOT found")

if os.path.exists(closed_path):
    print("Closed folder files:", os.listdir(closed_path))
else:
    print("Closed folder NOT found")

# ================= MODEL =================
class EyeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((24,24)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

# ================= DATA =================
try:
    train_data = ImageFolder(train_path, transform=transform)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    print("Classes:", train_data.class_to_idx)
except Exception as e:
    print("ERROR LOADING DATASET:", e)
    exit()

# ================= DEVICE =================
device = torch.device("cpu")

model = EyeCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ================= TRAIN =================
EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

# ================= SAVE =================
torch.save(model.state_dict(), "models/eye_cnn.pth")
print("Model saved!")