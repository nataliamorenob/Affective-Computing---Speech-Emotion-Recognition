import torch
import torch.nn as nn
import torch.nn.functional as F

class sDNN(nn.Module):
    def __init__(self, input_dim=13*128, num_classes=8, dropout_rate=0.4):
        super(sDNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 512)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.drop2 = nn.Dropout(dropout_rate * 0.75)  # slightly less dropout
        self.fc3 = nn.Linear(256, 128)
        self.drop3 = nn.Dropout(dropout_rate * 0.5)  # even less dropout
        self.fc4 = nn.Linear(128, num_classes)

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