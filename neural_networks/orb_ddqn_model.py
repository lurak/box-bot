import torch
from torch import nn


class OrbDDQNModel(nn.Module):
    def __init__(self, num_actions):
        super(OrbDDQNModel, self).__init__()
        self.num_actions = num_actions

        self.shared = nn.Sequential(
            nn.Linear(150, 512),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(512, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(512, self.num_actions)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        # x = self.layers(x)
        # x = x.view(x.shape[0], -1)
        # x = self.value(x)
        x = self.shared(x)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + advantage - advantage.mean()
