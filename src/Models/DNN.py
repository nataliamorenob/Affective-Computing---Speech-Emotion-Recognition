import torch
import torch.nn as nn
import torch.nn.functional as F

class sDNN(nn.Module):
    def __init__(self):
        super(sDNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(13 * 128, 512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 128)
        self.drop3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(128, 8)  # 8 emotion classes (0–7)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = F.relu(self.fc3(x))
        x = self.drop3(x)
        x = self.fc4(x)  # logits (no softmax)
        return x