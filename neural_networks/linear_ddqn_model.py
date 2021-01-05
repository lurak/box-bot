import torch
from torch import nn


class LinearDDQNModel(nn.Module):
    def __init__(self, num_actions):
        super(LinearDDQNModel, self).__init__()
        self.num_actions = num_actions

        self.value = nn.Sequential(
            nn.Linear(4, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(4, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        # x = self.layers(x)
        # x = x.view(x.shape[0], -1)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + advantage - advantage.mean()
