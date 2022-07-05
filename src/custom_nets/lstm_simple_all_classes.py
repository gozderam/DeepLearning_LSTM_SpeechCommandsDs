import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMSimpleAll(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(86, 64, 2, batch_first=True)
        self.fc1 = nn.Linear(64 * 64, 1024)
        self.fc2 = nn.Linear(1024, 31)
        self.name = 'LSTMSimpleAll'

    def forward(self, x):
        x = torch.squeeze(x, 1)
        x, _ = self.lstm(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x