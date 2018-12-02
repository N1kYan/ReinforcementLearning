from __future__ import print_function
import gym
import quanser_robots
import time
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from Discretization import QubeDiscretization
from Regression import Regressor
from DynamicProgramming import value_iteration
from DynamicProgramming import policy_iteration

# Create environment
env = gym.make("Qube-v0")

print("State space:  Shape:{}  Min:{}  Max:{} ".format(np.shape(env.observation_space), env.observation_space.low,
                                                       env.observation_space.high))
print("Action space:  Shape:{}  Min:{}  Max:{} ".format(np.shape(env.action_space), env.action_space.low,
                                                        env.action_space.high))

"""
    The Qube-v0 environment:

    The action space is a Box(1,) with values between [-15, 15] (joint effort)

    The state space is the current angle of theta and alpha in radians and their angular velocities
     min: -2, -pi, -30, 40; max:2, pi, 30, 40

"""

# Create Discretization and Regression objects

larry = QubeDiscretization(state_space_size=(8+1, 8+1, 6+1, 8+1), action_space_size=(30 + 1))
print(larry.state_space)
# print(larry.map_to_index([-1.7, 2]))
# print(larry.action_space)

reg = Regressor()

# Learning episodes / amount of samples for regression
epochs = 10000

# Perform regression
regressorState, regressorReward = reg.perform_regression(epochs, env)

# Perform dynamic programming to get value function and near optimal policy
value_function, policy = value_iteration(regressorState = regressorState, regressorReward = regressorReward, disc=larry,
                                         theta=0.1, gamma=0.7)
#value_function, policy = policy_iteration(regressorState=regressorState, regressorReward=regressorReward,
#                                          disc=larry, theta=0.1, gamma=0.5)

"""
    Evaluation stuff to see the predictions, discretizations and learned functions in action

"""
rewards_per_episode = []

episodes = 100

print("Evaluating...")

for e in range(episodes):

    # Discretize first state
    state = env.reset()
    index = larry.map_to_index(state)

    cumulative_reward = [0]

    for t in range(200):
        # Render environment
        env.render()

        # Do step according to policy and get observation and reward
        action = np.array([policy[index[0], index[1]]])

        state, reward, done, info = env.step(action)

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