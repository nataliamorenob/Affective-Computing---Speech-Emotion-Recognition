import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, n_features=56, time_steps=216, num_classes=8):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2)

        self.drop_conv = nn.Dropout(0.3)

        # compute flattened size dynamically
        dummy = torch.zeros(1, 1, n_features, time_steps)
        with torch.no_grad():
            out = self._forward_features(dummy)
            flat_dim = out.numel()

        self.fc1 = nn.Linear(flat_dim, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.drop_fc = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, num_classes)

    def _forward_features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.drop_conv(x)
        return x

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, n_features, time_steps)
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.drop_fc(x)
        return self.fc2(x)  # logits (no softmax)
