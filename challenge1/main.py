from __future__ import print_function
import gym
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from Discretization import Discretization
from Discretization import my_arctan
from Regression import Regressor
from DynamicProgramming import value_iteration
from DynamicProgramming import policy_iteration

# Create gym/quanser environment
env = gym.make('Pendulum-v0')

"""
    The Pendulum environment:

    The action space is a Box(1,) with values between [-2, 2] (joint effort)
    
    The state space is the cos, sin and velocity of the angle
    max:1,1,8; min:-1,-1,-8

"""

# Create Discretization and Regression objects

larry = Discretization("degree_only","Pendulum",state_space_size=(18+1, 16+1),action_space_size=17)
# print(larry.state_space)
# print(larry.action_space)
reg = Regressor()

# Learning episodes / amount of samples for regression
epochs = 10000

# Perform regression
reg.perform_regression(epochs, env)

# Perform dynamic programming to get value function and near optimal policy
value_function, policy = value_iteration(disc=larry, theta=0.001, gamma=0.1)
# value_function, policy = policy_iteration(larry, theta=1, gamma=0.1)

"""
    Evaluation stuff to see the predictions, discretizations and learned functions in action

"""
rewards_per_episode = []

episodes = 100

print("Evaluating...")

for e in range(episodes):

    # Discretize first state
    state = env.reset()
    state = np.array([my_arctan(state[0], state[1]), state[2]])
    index = larry.map_to_index(state)

    cumulative_reward = [0]

    for t in range(200):
        # Render environment
        env.render()

        # Do step according to policy and get observation and reward
        action = np.array([policy[index[0], index[1]]])

        # action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        state = np.array([my_arctan(observation[0], observation[1]), observation[2]])

        cumulative_reward.append(cumulative_reward[-1] + reward)

        # Discretize observed state
        index = larry.map_to_index(state)

        if done:
            print("Episode {} finished after {} timesteps".format(e + 1, t + 1))
            break

    rewards_per_episode.append(cumulative_reward)

print("...done")

# TODO: Look at calculation of mean cumulative rewards
# Average reward over episodes
rewards = np.average(rewards_per_episode, axis=0)

env.close()

# Plot rewards per timestep averaged over episodes
plt.figure()
plt.plot(rewards, label='Cumulative reward per timestep, averaged over {} episodes'.format(episodes))
plt.legend()
plt.show()