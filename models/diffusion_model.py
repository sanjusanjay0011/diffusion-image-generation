import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions

class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 64 * 64, 1024)
        self.fc2 = nn.Linear(1024, 3 * 256 * 256)  # Output image size is 256x256 RGB

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(x.size(0), 3, 256, 256)  # Output image size is 256x256
        return x
