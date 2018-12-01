import numpy as np
import gym
import quanser_robots
from quanser_robots.pendulum.pendulum_v2 import angle_normalize
"""
    True model for quanser Pendulum-v2
    
    Methods for state transition function and reward function.
    Both methods take an array of [angle, angular velocity] and [action]
    
"""


def transition(input):
    state = np.array([input[0], input[1]])
    action = input[2]
    g = 10.0
    l = 1.0
    m = 1.0
    dt = 0.05
    action = np.clip(action, -2.0, 2.0)
    newthdot = state[1] + (-3 * g / (2 * l) * np.sin(state[0] + np.pi) + 3. / (m * l ** 2) * action) * dt
    newth = state[0] + newthdot * dt
    newthdot = np.clip(newthdot, -8, 8)
    return np.array([newth, newthdot])


def reward(input):
    state = np.array([input[0], input[1]])
    action = input[2]
    action = np.clip(action, -2.0, 2.0)
    costs = - angle_normalize(state[0])**2 + .1 * state[1]**2 + .001 * (action**2)
    return costs

def main():
    env = gym.make("Pendulum-v2")
    state = env.reset()
    action = np.array([2.0])
    x = np.array([state[0], state[1], action])
    print("State: ", state)
    print("Action: ", action)

    print("Predicted state: ", transition(x))
    print("Predicted reward: ", reward(x))

    s, r, dones, infos = env.step(action)

    print("True state: ", s)
    print("True reward: ",r)
    print()

if __name__ == "__main__":
    main()
