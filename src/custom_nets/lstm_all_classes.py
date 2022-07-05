import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMAll(nn.Module):
    def __init__(self, dropout_p=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding='same')
        self.conv12 = nn.Conv2d(32, 32, 3, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=dropout_p)
        self.lstm = nn.LSTM(1376, 512, 2)
        self.fc1 = nn.Linear(16384, 1024)
        self.fc2 = nn.Linear(1024, 31)
        self.name = 'LSTMAll'

    def forward(self, x):
        x = self.pool(F.relu(self.conv12(F.relu(self.conv1(x)))))
        x = self.dropout(x)
        x = torch.flatten(x, 2)
        x, _ = self.lstm(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x