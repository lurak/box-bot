import gym
import json
from DDQN.ddqn_agent import DDQNAgent

env = gym.make('Boxing-v0')
env.reset()

with open("config_files/ddqn.json", "r") as f:
    params = json.load(f)

agent = DDQNAgent(**params)

agent.train(env)
