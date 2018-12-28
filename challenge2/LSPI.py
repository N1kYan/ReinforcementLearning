from __future__ import print_function
import sys
import gym
import quanser_robots
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


def pi(s, w):
    def policy_a(a):
        su = 0
        for (mist, w_dash) in zip(kleinvieh, w):
            dash = mist(s, a) * w_dash
            su += dash
        return su
    # return max((policy_a(a) for a in A))
    return np.array([A[np.argmax([policy_a(a) for a in A])]])
    # return np.array(max((policy_a(a) for a in A)))


def lstdq(D, phi, gamma, w):
    k = len(phi)
    A_tilde = np.zeros(shape=(k, k))
    b_tilde = np.zeros(shape=(k, 1))
    for (s, a, r, s_dash) in D:
        phis = np.array([p(s, a) for p in phi])
        phis_dash = np.array([p(s_dash, pi(s_dash, w)) for p in phi])
        A_tilde += phis*(phis-gamma*phis_dash).T
        b_tilde += phis*r
    return np.matmul(np.linalg.inv(A_tilde), b_tilde)


def lspi(D, phi, gamma, epsilon, w_0):
    w_dash = w_0
    while True:
        w = w_dash
        w_dash = lstdq(D, phi, gamma, w)
        diff = np.linalg.norm(x=(w-w_dash), ord=2)
        print("Diff:", diff)
        if diff < epsilon:
            break
    return w


def sample(epochs):
    D = []
    state = env.reset()
    for e in range(epochs):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        D.append((state, action, reward, next_state))
        state = np.copy(next_state)
        if done:
            state = env.reset()
    return D


def evaluate(w_star, episodes):
    plt.figure()
    plt.title("Cumulative reward per episode")
    for e in range(episodes):
        state = env.reset()
        rewards = []
        cumulative_reward = [0]
        while True:
            env.render()
            action = pi(state, w_star)
            # print(action)
            next_state, reward, done, info = env.step(action)
            rewards.append(reward)
            cumulative_reward.append(reward+cumulative_reward[-1])
            if done:
                break
            state = np.copy(next_state)
        plt.plot(cumulative_reward)
    # Average cumulative rewards over episodes
    plt.show()


def main():
    print("Sampling...", end='')
    sys.stdout.flush()
    D = sample(epochs=10000)
    print("done")
    # w_0 = np.ones(shape=(len(kleinvieh), 1))
    w_0 = np.zeros(shape=(len(kleinvieh), 1))
    gamma = 0.95
    epsilon = 1e-6
    print("Learning...")
    w_star = lspi(D, kleinvieh, epsilon, gamma, w_0)
    print("...done")
    print("Evaluating...", end='')
    sys.stdout.flush()
    evaluate(w_star, episodes=10)
    print("done")


# Define gym environment
env = gym.make("CartpoleStabShort-v0")
# Set discrete action space
A = np.linspace(start=env.action_space.low, stop=env.action_space.high, num=25)  # 49
print(env.action_space.sample())
print("Discrete Action Space: ", A)
# Define basis functions
"""
kleinvieh = [
    lambda s, a: s[0] * a,
    lambda s, a: s[1] * a,
    lambda s, a: s[2] * a,
    lambda s, a: s[3] * a,
    lambda s, a: s[4] * a,
]
"""
kleinvieh = [
    lambda s, a: s[0]*s[1]*s[2]*s[3]*s[4]*a,
    lambda s, a: s[0]*a**2,
    lambda s, a: s[1]*a**2,
    lambda s, a: s[2]*a**2,
    lambda s, a: s[3]*a**2,
    lambda s, a: s[4]*a**2,
]
if __name__ == '__main__':
    main()
