import torch
import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, n_mfcc=13, time_steps=128, hidden_dim=128, num_layers=2, num_classes=8, dropout_rate=0.3):
        super(LSTMNet, self).__init__()
        # Note: For LSTM, input_dim is the feature dimension (n_mfcc), not n_mfcc * time_steps
        self.lstm = nn.LSTM(n_mfcc, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Input shape: (batch, n_mfcc, time_steps) → we need (batch, time_steps, n_mfcc)
        x = x.permute(0, 2, 1)  # Now (batch, time_steps, n_mfcc)
        _, (h_n, _) = self.lstm(x)
        out = self.dropout(h_n[-1])  # Apply dropout to final hidden state
        out = self.fc(out)
        return out
