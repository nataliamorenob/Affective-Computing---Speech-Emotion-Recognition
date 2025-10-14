"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN_Attention(nn.Module):
    def __init__(self, n_mfcc=13, time_steps=128, num_classes=8, cnn_filters=64, lstm_hidden=128):
        super(CRNN_Attention, self).__init__()

        # CNN part
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, cnn_filters, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

        # Placeholder for LSTM (we'll init later)
        self.lstm = None
        self.lstm_hidden = lstm_hidden
        self.num_classes = num_classes

        # Attention & final classifier will also be set dynamically later
        self.attention = None
        self.fc = None

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, n_mfcc, time_steps)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Get shape
        b, c, f, t = x.size()

        # Prepare for LSTM: (batch, t, features)
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # Initialize LSTM dynamically the first time forward() runs
        if self.lstm is None:
            self.lstm = nn.LSTM(
                input_size=c * f,
                hidden_size=self.lstm_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            ).to(x.device)
            self.attention = nn.Linear(self.lstm_hidden * 2, 1).to(x.device)
            self.fc = nn.Linear(self.lstm_hidden * 2, self.num_classes).to(x.device)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Attention
        attn_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        weighted = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)

        out = self.fc(weighted)
        return out
    
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN_Attention(nn.Module):
    def __init__(self, n_mfcc=56, time_steps=216, num_classes=8, cnn_filters=64, lstm_hidden=128):
        super(CRNN_Attention, self).__init__()

        # CNN feature extractor (with stronger pooling)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, cnn_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(cnn_filters)
        self.pool = nn.MaxPool2d((2, 4))   # ↓ both freq and time dimensions

        # Compute reduced feature dimension dynamically
        dummy = torch.zeros(1, 1, n_mfcc, time_steps)
        with torch.no_grad():
            out = self._forward_cnn(dummy)
            _, c, f, t = out.shape
            self.lstm_input_dim = c * f

        # LSTM + Attention
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.attention = nn.Linear(lstm_hidden * 2, 1)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)
        self.dropout = nn.Dropout(0.4)

    def _forward_cnn(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        return x

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, n_features, time_steps)
        x = self._forward_cnn(x)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        lstm_out, _ = self.lstm(x)

        # Attention
        attn_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        weighted = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)

        out = self.dropout(weighted)
        return self.fc(out)
