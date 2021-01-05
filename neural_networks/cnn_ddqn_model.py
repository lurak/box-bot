import torch
from torch import nn


class CnnDDQNModel(nn.Module):
    def __init__(self, num_actions):
        super(CnnDDQNModel, self).__init__()
        self.num_actions = num_actions

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        self.hidden = nn.Sequential(
            nn.Linear(800, 256),
            nn.ReLU()
        )

        self.value = nn.Sequential(
            nn.Linear(256, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(256, self.num_actions)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.shape[0], -1)
        x = self.hidden(x)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + advantage - advantage.mean()
