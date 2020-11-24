from replay_buffer import ReplayBuffer
from QModel import QModel
import numpy as np
import gym


class QAgent:
    def __init__(self,
                 gamma,
                 action_number,
                 minibatch,
                 epsilon,
                 episodes,
                 begin_train
                 ):
        self.model = QModel((150, 100, 1), action_number)
        self.const_model = self.model.clone()
        self.episodes = episodes
        self.epsilon = epsilon
        self.minibatch = minibatch
        self.action_number = action_number
        self.gamma = gamma
        self.action = None
        self.state = None
        self.begin_train = begin_train

    def epsilon_greedy(self):
        if (1 - self.epsilon) <= np.random.uniform(0, 1):
            self.action = np.argmax(self.model.model.predict(self.state), axis=-1)
        else:
            self.action = np.random.randint(self.action_number)
        return self.action

    @staticmethod
    def preprocess_observation(observation):
        rgb = observation[30:180, 30: 130]
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def train(self, env: gym.wrappers.time_limit.TimeLimit):
        replay_buffer = ReplayBuffer()
        for i in range(self.episodes):
            action = self.epsilon_greedy()
            reward, next_observation, done, _ = env.step(action)
            next_state = self.preprocess_observation(next_observation)
            replay_buffer.push(action=action,
                               state=self.state,
                               reward=reward,
                               next_state=next_state,
                               done=done)
            o_act, o_state, o_reward, o_next_state, o_done = \
                replay_buffer.select(self.minibatch)
            q_next = self.const_model.model.predict(o_next_state)
            y_hat = o_reward + self.gamma*np.max(q_next, axis=-1) * (1 - o_done)
            self.model.model.fit(o_state, y_hat, epochs=1, verbose=2)



