from replay_buffer.replay_buffer import ReplayBuffer
import numpy as np
import gym
import torch
from time import time, sleep
from DQN.dqn_model import BoxModel
import matplotlib.pyplot as plt


class DQNAgent:
    def __init__(self,
                 gamma,
                 action_number,
                 minibatch,
                 episodes,
                 begin_train,
                 train_step,
                 begin_copy,
                 copy_step,
                 epsilon_delta,
                 epsilon_start,
                 epsilon_end,
                 load_model,
                 path_to_load,
                 path_to_save,
                 episode_steps,
                 episode_to_save,
                 max_buffer_len
                 ):

        # Epsilon

        self.epsilon_delta = epsilon_delta
        self.epsilon_end = epsilon_end
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start

        # Main Params

        self.minibatch = minibatch
        self.action_number = action_number
        self.gamma = gamma

        # Episode Params

        self.begin_train = begin_train
        self.begin_copy = begin_copy
        self.copy_step = copy_step
        self.train_step = train_step
        self.episodes = episodes
        self.episode_steps = episode_steps
        self.episode_to_save = episode_to_save

        # I/O params

        self.path_to_load = path_to_load
        self.path_to_save = path_to_save
        self.load_model = load_model

        # Model Fields

        self.action = None
        self.state = None
        self.replay_buffer = ReplayBuffer(max_buffer_len)

        # Model
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.model = BoxModel((150, 100, 1), action_number).to(self.device)
        if self.load_model:
            self.model.load_state_dict(torch.load(self.path_to_load))

        # Rewards

        self.rewards_white, self.rewards_black, self.rewards = [], [], []

    def reduce_epsilon(self, episode):
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1. * episode / self.epsilon_delta)

    def epsilon_greedy(self):
        if (1 - self.epsilon) <= np.random.random():
            self.action = np.random.randint(self.action_number)
        else:
            state = torch.autograd.Variable(torch.FloatTensor(self.state).to(self.device).unsqueeze(0))
            self.action = self.model(state).max(1)[1].item()
        return self.action

    @staticmethod
    def preprocess_observation(observation):
        rgb = observation[30:180, 30: 130] / 255
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray.reshape(1, 150, 100)

    def transition_process(self, o_state, o_act, o_reward, o_next_state, o_done):

        return \
            torch.autograd.Variable(torch.FloatTensor(np.float32(o_state)).to(self.device)), \
            torch.autograd.Variable(torch.LongTensor(o_act).to(self.device)), \
            torch.autograd.Variable(torch.FloatTensor(o_reward).to(self.device)), \
            torch.autograd.Variable(torch.FloatTensor(np.float32(o_next_state)).to(self.device)), \
            torch.autograd.Variable(torch.FloatTensor(o_done).to(self.device))

    def train_model(self):
        o_state, o_act, o_reward, o_next_state, o_done = \
            self.transition_process(*self.replay_buffer.sample(self.minibatch))
        q = self.model(o_state)
        q_next = self.model(o_next_state)
        y_hat = o_reward + self.gamma * q_next.max(1)[0] * (1 - o_done)
        loss = (q.gather(1, o_act.unsqueeze(1)).squeeze(1) - torch.autograd.Variable(y_hat.data)).pow(2).mean()
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

    def print(self, episode, reward_black,
              reward_white, epsilon):
        print(f"For episode {episode} reward white - "
              f"{reward_white} and black - {reward_black},"
              f"epsilon - {epsilon}")

    def train(self, env: gym.wrappers.time_limit.TimeLimit):
        start = time()
        print("Begin to Train")

        for episode in range(self.episodes):
            observation = env.reset()
            self.state = self.preprocess_observation(observation)
            reward_black, reward_white, total_reward = 0, 0, 0
            for episode_steps in range(self.episode_steps):
                action = self.epsilon_greedy()
                next_observation, reward, done, _ = env.step(action)
                reward_black += (reward < 0) * abs(reward)
                reward_white += (reward > 0) * reward
                total_reward += reward
                next_state = self.preprocess_observation(next_observation)
                self.replay_buffer.push(self.state, action, reward, next_state, done)
                if len(self.replay_buffer) >= self.begin_train:
                    self.train_model()
                # if (episode_step >= self.begin_copy) and (episode_step % self.copy_step == 0):
                #     plt.plot(total_reward)
                #     plt.show()
                    # self.const_model = self.model.clone()
                if done:
                    break

            self.reduce_epsilon(episode)
            if episode != 0 and episode % self.episode_to_save == 0:
                torch.save(self.model.state_dict(), self.path_to_save)
                plt.plot(self.rewards)
                plt.show()

            self.rewards_black.append(reward_black)
            self.rewards_white.append(reward_white)
            self.rewards.append(total_reward)
            self.print(episode, reward_black=reward_black,
                       reward_white=reward_white, epsilon=self.epsilon)
            print(time() - start)

    def play(self, env: gym.wrappers.time_limit.TimeLimit):
        observation = env.reset()
        reward_black, reward_white, total_reward = 0, 0, 0
        for episode_steps in range(self.episode_steps):
            state = self.preprocess_observation(observation)
            state = torch.autograd.Variable(torch.FloatTensor(state).to(self.device).unsqueeze(0))
            print(self.model(state))
            action = self.model(state).max(1)[1].item()
            observation, reward, done, _ = env.step(action)
            reward_black += (reward < 0) * abs(reward)
            reward_white += (reward > 0) * reward
            total_reward += reward
            sleep(0.01)
            env.render()
            if done:
                break
        print(total_reward)




