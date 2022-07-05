import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCNNAllNoPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, (4, 1), padding='same')
        self.conv2 = nn.Conv2d(4, 8, (4, 1), padding='same')
        self.conv3 = nn.Conv2d(8, 16, (4, 1), padding="same")

        self.lstm1 = nn.LSTM(1376, 1024, 1, batch_first=True)
        self.lstm2 = nn.LSTM(1024, 512, 1, batch_first=True)
        self.lstm3 = nn.LSTM(512, 256, 1, batch_first=True)

        self.fc1 = nn.Linear(16384, 4096)
        self.fc2 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512, 31)
        self.name = 'LSTMCNNAllNoPooling'

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.permute(0, 2, 3, 1)
        x = torch.flatten(x, 2)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x