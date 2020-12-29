import gym
import json
from DQN.dqn_agent import DQNAgent

env = gym.make('Boxing-v0')
env.reset()

with open("config_files/dqn.json", "r") as f:
    params = json.load(f)

agent = DQNAgent(**params)

agent.train(env)
