from tqdm import trange

from replay_buffer.replay_buffer import ReplayBuffer
import numpy as np
import gym
import torch
from time import sleep
from skimage.transform import resize
from skimage.color import rgb2gray
# from cart_pole_model import DDQNModel
import matplotlib.pyplot as plt
from GeneralModel.genetal_agent import GeneralAgent


class DDQNAgentCnn(GeneralAgent):
    def __init__(self,
                 gamma,
                 action_number,
                 minibatch,
                 episodes,
                 begin_train,
                 copy_step,
                 epsilon_delta,
                 epsilon_start,
                 epsilon_end,
                 load_model,
                 path_to_load,
                 path_to_save,
                 plots_to_save,
                 episode_steps,
                 episode_to_save,
                 max_buffer_len,
                 model_type
                 ):

        super().__init__(gamma=gamma,
                         action_number=action_number,
                         path_to_load=path_to_load,
                         path_to_save=path_to_save,
                         plots_to_save=plots_to_save,
                         load_model=load_model,
                         episode_to_save=episode_to_save,
                         episodes=episodes,
                         model_type=model_type)
        # Epsilon

        self.epsilon_delta = epsilon_delta
        self.epsilon_end = epsilon_end
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start

        # Main Params

        self.minibatch = minibatch

        # Episode Params

        self.begin_train = begin_train
        self.copy_step = copy_step
        self.episode_steps = episode_steps

        # Model Fields

        self.action = None
        self.state = None
        self.replay_buffer = ReplayBuffer(max_buffer_len)

        # Model
        self.target_model = model_type(action_number).to(self.device)
        self.update_target()

        # Rewards

        self.rewards_white, self.rewards_black, self.rewards = [], [], []
        self.losses = []
        self.periodic_reward = 0
        self.periodic_rewards = []

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

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
    def preprocess_observation(obs):
        img = resize(rgb2gray(obs[0:188, 23:136, :]), (28, 28), mode='constant')
        img = img.reshape(1, 28, 28)
        return img

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
        q = self.model(o_state).gather(1, o_act.unsqueeze(1)).squeeze(1)
        q_next = self.target_model(o_next_state)
        y_hat = o_reward + self.gamma * q_next.max(1)[0] * (1 - o_done)
        loss = (q - y_hat.detach()).pow(2).mean()

        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()
        return loss

    def init_new_episode(self, env):
        observation = env.reset()
        self.state = self.preprocess_observation(observation)

    def episode_check(self, episode, loss):

        if episode % self.copy_step == 0:
            self.losses.append(loss)
            self.update_target()

        if episode % self.episode_steps == 0:
            self.periodic_rewards.append(self.periodic_reward / self.episode_steps)
            self.periodic_reward = 0

        if episode % self.episode_to_save == 0:
            torch.save(self.model.state_dict(), self.path_to_save)
            fig = plt.figure()
            plt.plot(self.rewards)
            fig.savefig(self.plots_to_save + '_reward.png')
            plt.close(fig)
            fig = plt.figure()
            plt.plot(self.losses)
            fig.savefig(self.plots_to_save + '_loss.png')
            plt.close(fig)
            fig = plt.figure()
            plt.plot(self.periodic_rewards)
            fig.savefig(self.plots_to_save + '_periodic_reward.png')
            plt.close(fig)
            
    def train(self, env: gym.wrappers.time_limit.TimeLimit):
        self.init_new_episode(env)
        total_reward = 0
        episode_reward = 0
        loss = 0
        for episode in self.trangle:
            self.trangle.set_description(
                f"Episode: {episode} | Episode Reward {episode_reward} | Periodic reward "
                f"{self.periodic_reward / self.episode_steps} | Average Reward {total_reward / (episode + 1)}"
            )
            self.trangle.refresh()
            action = self.epsilon_greedy()
            next_observation, reward, done, _ = env.step(action)
            total_reward += reward
            episode_reward += reward
            self.periodic_reward += reward
            next_state = self.preprocess_observation(next_observation)
            self.replay_buffer.push(self.state, action, reward, next_state, done)
            self.state = next_state
            if len(self.replay_buffer) >= self.begin_train:
                loss = self.train_model()

            self.reduce_epsilon(episode)
            self.episode_check(episode, loss)

            if done:
                self.init_new_episode(env)
                self.rewards.append(episode_reward)
                episode_reward = 0




