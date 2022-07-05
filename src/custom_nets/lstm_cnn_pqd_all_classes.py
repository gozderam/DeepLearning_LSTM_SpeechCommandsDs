from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCNNAllpqd(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding='same')
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding="same")
        self.pool = nn.MaxPool2d(2, 2)
        self.lstm = nn.LSTM(640, 100, 1, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2 * 100 * 8, 256)
        self.fc2 = nn.Linear(256, 31)
        self.name = 'LSTMCNNAllpqd'

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.permute(0, 2, 3, 1)
        x = torch.flatten(x, 2)
        x, _ = self.lstm(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x