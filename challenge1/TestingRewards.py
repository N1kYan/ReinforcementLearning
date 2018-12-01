from __future__ import print_function
import gym
import quanser_robots
import time
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from Discretization import PendulumDiscretization
from Discretization import EasyPendulumDiscretization
from Regression import Regressor
from DynamicProgramming import value_iteration
from DynamicProgramming import policy_iteration
from Utils import *

def testing_rewards(env):
    env.reset()

    # observation space is 3d angle of pendulum cos, sin, velocity max:1,1,8; min:-1,-1,-8
    print("Observation low:", env.observation_space.low)
    print("Observation high:", env.observation_space.high)

    # action space is a Box(1,) with values between [-2,2], joint effort
    print("Action low:", env.action_space.low)
    print("Action high: ", env.action_space.high)


    # # reward formular: -(theta^2 + 0.1*theta_dt^2 + 0.001*action^2) (-16.27 is worst, 0 best)
    # base_state = env.reset()
    # print("Base state: ", base_state)
    # a = env.action_space.sample()
    # print("Take action: ", a)
    # state, reward, done, info = env.step(a)
    # # state = [angle, velocity]
    # print("Result: ", state, reward, done, info)
    #
    # a_rev = [-a[0]]
    # base_state_vel_rev = - base_state[1]
    # print(a_rev)
    # print("State after reverse: ", env.step([- state[1]]))

    for i in range(50):
        base_state = env.reset()
        a = env.action_space.sample()
        state, reward, done, info = env.step(a)

        print("Start: ", base_state, ", Action: ", a, ", End: ", state, ", Reward: ", reward)

        reward_calc(base_state, a, state)


def reward_calc(start, action, end):
    m = 1
    l = 1
    g = 10

    th = start[0]
    thdot = start[1]
    u = action[0]
    dt = 0.05

    newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (
                m * l ** 2) * u) * dt
    newth = th + newthdot*dt

    state_recov = np.array([newth, newthdot])
    print("Recovered State: ", state_recov, " vs. True State: ", end)


# Create gym environment
env = gym.make('Pendulum-v2')
testing_rewards(env)