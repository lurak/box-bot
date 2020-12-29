import torch
from torch import nn


class DDQNModel(nn.Module):
    def __init__(self, num_actions):
        super(DDQNModel, self).__init__()
        self.num_actions = num_actions

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.value = nn.Sequential(
            nn.Linear(8640, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(8640, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.shape[0], -1)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + advantage - advantage.mean()
