import torch
from torch import nn
import numpy as np


class BoxModel(nn.Module):
    def __init__(self, num_actions):
        super(BoxModel, self).__init__()
        self.num_actions = num_actions

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(8640, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        out = self.layers(x)
        out = out.view(x.shape[0], -1)
        out = self.fc(out)
        return out

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(torch.autograd.Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob