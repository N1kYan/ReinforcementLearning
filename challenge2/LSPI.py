from __future__ import print_function
import sys
import gym
import quanser_robots
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


# Policy returning the linear combination of basis functions and weights
def pi(s, w,):
    def policy_a(a):
        su = 0
        for (mist, w_dash) in zip(kleinvieh, w):
            dash = mist(s, a) * w_dash
            su += dash
        return su
    # return max((policy_a(a) for a in A))
    return np.array([A[np.argmax([policy_a(a) for a in A])]])
    # return np.array(max((policy_a(a) for a in A)))


# Optimized LSTDQ
def lstdq_opt(D, phi,gamma, w):
    k = len(phi)
    sigma = 100
    B_tilde = np.eye(N=k, M=k)*(1/sigma)
    b_tilde = np.zeros(shape=(k, 1))
    for (s, a, r, s_dash) in D:
        phis = np.array([p(s, a) for p in phi]).reshape(-1, 1)
        phis_dash = np.array([p(s_dash, pi(s_dash, w)) for p in phi]).reshape(-1, 1)
        B_tilde -= (np.matmul(B_tilde, np.matmul(phis, np.matmul((phis-gamma*phis_dash).T, B_tilde)))
                    / (1+np.matmul((phis-gamma*phis_dash).T, np.matmul(B_tilde, phis))))
        b_tilde = b_tilde + phis*r
    return np.matmul(B_tilde, b_tilde)


# Least-Squares Temporal-Difference Q-Learning
def lstdq(D, phi, gamma, w):
    k = len(phi)
    A_tilde = np.zeros(shape=(k, k))
    b_tilde = np.zeros(shape=(k, 1))
    for (s, a, r, s_dash) in D:
        phis = np.array([p(s, a) for p in phi]).reshape(-1, 1)
        phis_dash = np.array([p(s_dash, pi(s_dash, w)) for p in phi])
        A_tilde = A_tilde + phis*(phis-gamma*phis_dash).T
        b_tilde = b_tilde + phis*r
    return np.matmul(np.linalg.inv(A_tilde), b_tilde)


# Least-Squares Policy Iteration
def lspi(D, phi, gamma, epsilon, w_0):
    w_dash = w_0
    while True:
        w = w_dash
        w_dash = lstdq(D=D, phi=phi, gamma=gamma, w=w)
        # w_dash = lstdq_opt(D=D, phi=phi, gamma=gamma, w=w)
        diff = np.linalg.norm(x=(w-w_dash), ord=2)
        print("Diff {} < {}: {} ".format(diff, epsilon, diff < epsilon))
        if diff < epsilon:
            break
    return w_dash  # w?


def sample(epochs):
    D = []
    state = env.reset()
    for e in range(epochs):
        # Random sample
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        D.append((state, action, reward, next_state))
        state = np.copy(next_state)
        if done:
            state = env.reset()
    return D


def evaluate(w_star, episodes, render, plot):
    if plot:
        plt.figure()
        # plt.title("Cumulative reward per episode")
        plt.title("Rewards per episode")
    for e in range(episodes):
        state = env.reset()
        rewards = []
        cumulative_reward = [0]
        chosen_actions = []
        time_step = 0
        while True:
            time_step += 1
            if render:
                env.render()
            action = pi(state, w_star)
            chosen_actions.append(action)
            # print(action)
            next_state, reward, done, info = env.step(action)
            rewards.append(reward)
            cumulative_reward.append(reward+cumulative_reward[-1])
            if done:
                print("Episode {} terminated after {} timesteps with a total cumulative reward of {}".
                      format(e, time_step, cumulative_reward[-1]))
                break
            state = np.copy(next_state)
        if plot:
            plt.plot(rewards)
            # plt.plot(cumulative_reward)
    # Average cumulative rewards over episodes
    if plot:
        plt.figure()
        plt.title("Distribution of chosen actions")
        plt.hist(x=chosen_actions, bins=len(A))
        plt.show()
    return cumulative_reward[-1]


def main():
    print("Sampling...", end='')
    sys.stdout.flush()
    D = sample(epochs=10000)
    print("done")
    # Initial weights
    w_0 = np.ones(shape=(len(kleinvieh), 1))
    for i in range(len(w_0)):
       w_0[i] = np.random.rand()**3
    # w_0.fill(20)
    # w_0 = np.zeros(shape=(len(kleinvieh), 1))
    gamma = 0.99  # 0.95  # 0.2
    # epsilon = 1e-2
    epsilon = 1  # should be reduced to perform better but results in super slow convergence
    print("Learning...")
    w_star = lspi(D=D, phi=kleinvieh, gamma=gamma, epsilon=epsilon, w_0=w_0)
    print("w*:", w_star)
    print("...done")
    print("Evaluating...", end='')
    sys.stdout.flush()
    evaluate(w_star, episodes=25, render=False, plot=True)
    print("done")


# Define gym environment
env = gym.make("CartpoleStabShort-v0")
print("State Space Low:", env.observation_space.low)
print("State Space Hight:", env.observation_space.high)
# Set discrete action space
A = np.linspace(start=env.action_space.low, stop=env.action_space.high, num=7)  # 13, 25, 49
print("Discrete Action Space: ", A)
# Define basis functions

kleinvieh = [
    # lambda s, a: s[0]*s[1]*s[2]*s[3]*s[4]*a,
    lambda s, a: 1,
    # lambda s, a: a,
    lambda s, a: np.abs(np.sin(s[0])) * a,
    lambda s, a: np.abs(np.sin(s[1])) * a,
    # lambda s, a: np.abs(np.sin(s[2])) * a,
    # lambda s, a: - np.abs(np.sin(s[2])) * a,

    # lambda s, a: np.sin(s[3])**2 * a,
    # lambda s, a: np.sin(s[4])**2 * a,
    # lambda s, a: np.exp(s[0]) * a,
    # lambda s, a: np.exp(s[1]) * a,
    # lambda s, a: np.exp(s[2]) * a,
    # lambda s, a: np.exp(s[3]) * a,
    # lambda s, a: np.exp(s[4]) * a,
    # lambda s, a: np.abs(np.cos(s[0])) * a,
    # lambda s, a: np.abs(np.cos(s[1])) * a,
    lambda s, a: np.abs(np.cos(s[2])) * a,
    # lambda s, a: np.cos(s[3])**2 * a,
    # lambda s, a: np.cos(s[4])**2 * a,
    lambda s, a: s[3]*np.abs(a),
    lambda s, a: s[4]*np.abs(a),

]

if __name__ == '__main__':
    main()
