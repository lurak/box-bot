from GeneralModel.genetal_agent import GeneralAgent
import gym
import torch
import numpy as np


class ReinforceAgent(GeneralAgent):
    def __init__(self,
                 gamma,
                 action_number,
                 episodes,
                 max_episode_step,
                 load_model,
                 path_to_load,
                 path_to_save,
                 plot_to_save,
                 episode_to_save,
                 model_type
                 ):
        super().__init__(gamma=gamma,
                         action_number=action_number,
                         path_to_load=path_to_load,
                         path_to_save=path_to_save,
                         load_model=load_model,
                         episode_to_save=episode_to_save,
                         episodes=episodes,
                         plots_to_save=plot_to_save,
                         model_type=model_type
                         )

        self.numsteps = []
        self.avg_numsteps = []
        self.all_rewards = []
        self.max_episode_steps = max_episode_step

    def update_policy(self, rewards, log_probs):
        discounted_rewards = []

        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + self.gamma ** pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                discounted_rewards.std() + 1e-9)  # normalize discounted rewards

        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

        self.model.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.model.optimizer.step()

    def train(self, env: gym.wrappers.time_limit.TimeLimit):
        total_reward = 0
        episode_reward = 0
        for episode in self.trangle:

            self.trangle.set_description(
                f"Episode: {episode} | Episode Reward {episode_reward} | | Average Reward {total_reward / (episode + 1)}"
            )

            self.save_all(episode)
            state = env.reset()
            log_probs = []
            rewards = []
            episode_reward = 0

            for steps in range(self.max_episode_steps):
                state = self.preprocess_observation(state)
                action, log_prob = self.model.get_action(state)
                new_state, reward, done, _ = env.step(action)
                total_reward += reward
                log_probs.append(log_prob)
                rewards.append(reward)
                episode_reward += reward
                if done:
                    self.update_policy(rewards, log_probs)
                    self.numsteps.append(steps)
                    self.avg_numsteps.append(np.mean(self.numsteps[-10:]))
                    self.all_rewards.append(np.sum(rewards))
                    self.rewards.append(episode_reward)
                    break
                state = new_state
