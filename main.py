import gym
import time
import matplotlib.pyplot as plt
from QAgent import QAgent

env = gym.make('Boxing-v0')
env.reset()

# Params
gamma = 0.95
action_number = 18
minibatch = 48
epsilon = 1
episodes = 100
begin_train = 50
train_step = 1
begin_copy = 100
copy_step = 100
epsilon_delta = epsilon / (episodes // 2)
min_epsilon = 0.00
#

agent = QAgent(
    gamma=gamma,
    action_number=action_number,
    minibatch=minibatch,
    epsilon=epsilon,
    episodes=episodes,
    begin_train=begin_train,
    train_step=train_step,
    begin_copy=begin_copy,
    copy_step=copy_step,
    epsilon_delta=epsilon_delta,
    min_epsilon=min_epsilon
)

agent.train(env)
env.close()