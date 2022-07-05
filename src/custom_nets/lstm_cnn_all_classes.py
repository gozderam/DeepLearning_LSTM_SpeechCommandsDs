import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCNNAll(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, (4, 1), padding='same')
        self.conv2 = nn.Conv2d(4, 8, (4, 1), padding='same')
        self.conv3 = nn.Conv2d(8, 16, (4, 1), padding="same")
        self.pool = nn.MaxPool2d(2, 2)

        self.lstm = nn.LSTM(160, 128, 2, batch_first=True)
        self.fc1 = nn.Linear(128 * 8, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 31)
        self.name = 'LSTMCNNAll'

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.permute(0, 2, 3, 1)
        x = torch.flatten(x, 2)
        x, _ = self.lstm(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x