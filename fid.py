import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from scipy.linalg import sqrtm

# Frechet Inception Distance (FID)
# Kernel Inception Distance (KID)
# Perceptual Hashing
# Structural Similarity Index (SSIM)
# Multi-Scale Structural Similarity Index (MS-SSIM)
# Learned Perceptual Image Patch Similarity (LPIPS)
# Deep Image Prior (DIP) with a Pretrained Network and a Loss Function

# Define the network (e.g., a simple CNN pretrained on MNIST)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load the pretrained network and set to evaluation mode
model = SimpleCNN()
# Assume model is pretrained on MNIST
model.eval()

# Function to extract features
def extract_features(images, model, device='cpu'):
    model.to(device)
    features = []
    with torch.no_grad():
        for img in images:
            img = img.to(device).unsqueeze(0)  # add batch dimension
            feat = model.features(img).view(-1).cpu().numpy()
            features.append(feat)
    return np.array(features)

# Load datasets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

reference_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
degraded_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

reference_images = [img for img, _ in reference_dataset]
degraded_images = [img for img, _ in degraded_dataset]

# Extract features
reference_features = extract_features(reference_images, model)
degraded_features = extract_features(degraded_images, model)

# Compute FID
mu1, sigma1 = np.mean(reference_features, axis=0), np.cov(reference_features, rowvar=False)
mu2, sigma2 = np.mean(degraded_features, axis=0), np.cov(degraded_features, rowvar=False)
fid = np.sum((mu1 - mu2)**2) + np.trace(sigma1 + sigma2 - 2 * sqrtm(sigma1.dot(sigma2)))

print('FID:', fid)
