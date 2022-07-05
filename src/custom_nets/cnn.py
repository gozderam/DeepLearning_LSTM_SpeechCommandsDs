import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNAll(nn.Module):
    def __init__(self, dropout_p=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding='same')
        self.conv12 = nn.Conv2d(32, 32, 3, padding='same')
        self.conv2 = nn.Conv2d(32, 64, 3, padding='same')
        self.conv22 = nn.Conv2d(64, 64, 3, padding='same')
        self.conv3 = nn.Conv2d(64, 128, 3, padding='same')
        self.conv32 = nn.Conv2d(128, 128, 3, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(10240, 512)
        self.fc2 = nn.Linear(512, 31)
        self.name = 'CNNAll'

    def forward(self, x):
        x = self.pool(F.relu(self.conv12(F.relu(self.conv1(x)))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv22(F.relu(self.conv2(x)))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv32(F.relu(self.conv3(x)))))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CNN3Classes(nn.Module):
    def __init__(self, dropout_p=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding='same')
        self.conv12 = nn.Conv2d(32, 32, 3, padding='same')
        self.conv2 = nn.Conv2d(32, 64, 3, padding='same')
        self.conv22 = nn.Conv2d(64, 64, 3, padding='same')
        self.conv3 = nn.Conv2d(64, 128, 3, padding='same')
        self.conv32 = nn.Conv2d(128, 128, 3, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(10240, 512)
        self.fc2 = nn.Linear(512, 3)
        self.name = 'CNN3Classes'

    def forward(self, x):
        x = self.pool(F.relu(self.conv12(F.relu(self.conv1(x)))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv22(F.relu(self.conv2(x)))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv32(F.relu(self.conv3(x)))))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CNN10Classes(nn.Module):
    def __init__(self, dropout_p=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding='same')
        self.conv12 = nn.Conv2d(32, 32, 3, padding='same')
        self.conv2 = nn.Conv2d(32, 64, 3, padding='same')
        self.conv22 = nn.Conv2d(64, 64, 3, padding='same')
        self.conv3 = nn.Conv2d(64, 128, 3, padding='same')
        self.conv32 = nn.Conv2d(128, 128, 3, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(10240, 512)
        self.fc2 = nn.Linear(512, 10)
        self.name = 'CNN10Classes'

    def forward(self, x):
        x = self.pool(F.relu(self.conv12(F.relu(self.conv1(x)))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv22(F.relu(self.conv2(x)))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv32(F.relu(self.conv3(x)))))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x