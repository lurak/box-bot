from replay_buffer import ReplayBuffer
from QModel import QModel
import numpy as np
import gym
import matplotlib.pyplot as plt

class QAgent:
    def __init__(self,
                 gamma,
                 action_number,
                 minibatch,
                 epsilon,
                 episodes,
                 begin_train,
                 train_step,
                 begin_copy,
                 copy_step,
                 epsilon_delta,
                 min_epsilon
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
        self.begin_copy = begin_copy
        self.copy_step = copy_step
        self.train_step = train_step
        self.min_epsilon = min_epsilon
        self.epsilon_delta = epsilon_delta

    def reduce_epsilon(self):
        self.epsilon -= self.epsilon_delta
        self.epsilon = max(self.epsilon, self.min_epsilon)

    def epsilon_greedy(self):
        if (1 - self.epsilon) <= np.random.uniform(0, 1):
            self.action = np.random.randint(self.action_number)
        else:
            self.action = np.argmax(self.model.model.predict(np.array([self.state])), axis=-1)
        return self.action

    def init_state(self, observation):
        self.state = self.preprocess_observation(observation)

    @staticmethod
    def preprocess_observation(observation):
        rgb = observation[30:180, 30: 130]
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray.reshape(gray.shape[0], gray.shape[1], 1)

    def train(self, env: gym.wrappers.time_limit.TimeLimit):
        self.rewards_white = []
        self.rewards_black = []
        self.rewards = []
        replay_buffer = ReplayBuffer()
        for i in range(self.episodes):
            observation = env.reset()
            self.init_state(observation)
            episode_step = 0
            reward_black = 0.
            reward_white = 0.
            total_reward = 0.
            self.reduce_epsilon()
            print(self.epsilon)
            while True:
                episode_step += 1
                action = self.epsilon_greedy()
                next_observation, reward, done, _ = env.step(action)
                reward_black += (reward < 0) * abs(reward)
                reward_white += (reward > 0) * reward
                total_reward += reward
                next_state = self.preprocess_observation(next_observation)
                replay_buffer.push(action=action,
                                   state=self.state,
                                   reward=reward,
                                   next_state=next_state,
                                   done=done)
                if (episode_step >= self.begin_train) and (episode_step % self.train_step == 0):
                    o_act, o_state, o_reward, o_next_state, o_done = \
                        replay_buffer.select(self.minibatch)
                    # print(o_next_state.shape)
                    q_next = self.const_model.model.predict(o_next_state)
                    y_hat = o_reward + self.gamma*np.max(q_next, axis=-1) * (1 - o_done)
                    self.model.model.fit(o_state, y_hat, epochs=1, verbose=0)
                if (episode_step >= self.begin_copy) and (episode_step % self.copy_step == 0):
                    print("WEOKFPPWEOKF")
                    self.const_model = self.model.clone()
                if done or episode_step >= 1000:
                    break
            self.rewards_black.append(reward_black)
            self.rewards_white.append(reward_white)
            self.rewards.append(total_reward)
            # if i % 500:
            print(reward_black)
            print(reward_white)
            print(total_reward)



