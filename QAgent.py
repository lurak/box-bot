from replay_buffer import ReplayBuffer
from QModel import QModel
import numpy as np


class QAgent:
    def __init__(self, epsilon, episodes):
        self.model = QModel((150, 100, 1), 18)
        self.episodes = episodes
        self.epsilon = epsilon
        self.action = None
        self.state = None

    def epsilon_greedy(self):
        if (1 - self.epsilon) <= np.random.uniform(0, 1):
            return self.action
        else:
            return self.action

    def train(self):
        replay_buffer = ReplayBuffer()
        for i in range(self.episodes):
            action = self.epsilon_greedy()

