import torch
import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=128, num_layers=2, num_classes=8):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Input shape: (batch, 13, 128) → we need (batch, time, features)
        x = x.permute(0, 2, 1)  # Now (batch, 128, 13)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out
