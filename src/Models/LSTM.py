import torch
import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, n_mfcc, time_steps, num_classes, hidden_dim=128, num_layers=2):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(
            input_size=n_mfcc,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch, n_mfcc, time_steps)
        x = x.permute(0, 2, 1)  # (batch, time_steps, n_mfcc)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out
