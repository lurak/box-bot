import random
import numpy as np


class ReplayBuffer:
    def __init__(self):
        self.history = []

    def push(self, action, state, reward, next_state, done):
        self.history.append((action, np.expand_dims(state, 0), reward,
                             np.expand_dims(next_state, 0), done))

    def select(self, batch_size):
        action, state, reward, next_state, done = zip(*random.sample(self.history, batch_size))
        return np.array(action), np.concatenate(state), np.array(reward), np.concatenate(next_state), np.array(done)
