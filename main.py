import gym
import time
import matplotlib.pyplot as plt

def preprocess_image(observation):
    rgb = observation[30:180, 30: 130]
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

env = gym.make('Boxing-v0')
env.reset()
# for _ in range(1000):
#     # env.render()
#     observation, reward, done, _ = env.step(env.action_space.sample()) # take a random action
#
#     # plt.imshow()
#     print(preprocess_image(observation).shape)
#     plt.show()
#     # print(reward)
#     time.sleep(0.3)
print(type(env))
env.close()