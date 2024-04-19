import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.maxPool = nn.MaxPool2d(2,2)

    def extract_features(self, x):
        x = self.maxPool(F.relu(self.conv1(x)))
        x = self.maxPool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return x
    
    def extract_conv_features(self, x):
        x = self.maxPool(F.relu(self.conv1(x)))
        x = self.maxPool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x):
        x = self.extract_features(x)
        x = self.fc3(x)
        return x

