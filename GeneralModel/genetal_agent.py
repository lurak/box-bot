import torch
from tqdm import trange
import matplotlib.pyplot as plt


class GeneralAgent:
    def __init__(self,
                 gamma,
                 action_number,
                 episodes,
                 load_model,
                 path_to_load,
                 path_to_save,
                 plots_to_save,
                 model_type,
                 episode_to_save,
                 ):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.episodes = episodes
        self.trangle = trange(self.episodes, desc='Training', leave=True)
        self.gamma = gamma
        self.action_number = action_number

        # I/O Params

        self.path_to_load = path_to_load
        self.path_to_save = path_to_save
        self.load_model = load_model
        self.episode_to_save = episode_to_save
        self.plots_to_save = plots_to_save

        # Metrics

        self.rewards_white, self.rewards_black, self.rewards = [], [], []

        # Model

        self.model = model_type(action_number).to(self.device)
        if self.load_model:
            self.model.load_state_dict(torch.load(self.path_to_load))

    def save_all(self, episode):
        if episode % self.episode_to_save == 0:
            torch.save(self.model.state_dict(), self.path_to_save)
            fig = plt.figure()
            plt.plot(self.rewards)
            fig.savefig('plots/plt_reward.png')
            plt.close(fig)

    @staticmethod
    def preprocess_observation(observation):
        rgb = observation[30:180, 30: 130] / 255
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray.reshape(1, 150, 100)