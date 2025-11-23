
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# This script trains a LeNet-like CNN on MNIST and evaluates robustness using
# a multi-step PGD (Projected Gradient Descent) adversarial attack.
# PGD iteratively applies small FGSM-like steps and projects back into an
# epsilon-ball around the original image, providing a stronger attack.

# ---------------------------
# 3. LeNet model
# ---------------------------
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # MNIST is 1 x 28 x 28
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)   # -> 6 x 24 x 24
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # -> 16 x 8 x 8 (after pooling)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)  # 6 x 12 x 12

        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)  # 16 x 4 x 4

        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


