import gym
import json
from REINFORCE.reinforce_agent import ReinforceAgent

env = gym.make('Boxing-v0')
env.reset()

with open("config_files/reinforce.json", "r") as f:
    params = json.load(f)

agent = ReinforceAgent(**params)

agent.train(env)
