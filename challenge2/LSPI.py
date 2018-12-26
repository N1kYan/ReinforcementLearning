import gym
import quanser_robots
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartpoleStabShort-v0")


# Basis function
kleinvieh = [
    lambda s, a: s*np.log(a),
    lambda s, a: np.log(s)*a,
]


def pi(s, w):
    return argmax a kleinvieh(s, a)*w

def lstdq(D, phi, gamma, w):
    k = len(phi)
    A_tilde = np.zeros(shape=(k, k))
    b_tilde = np.zeros(shape=(k, 1))
    for (s, a, r, s_dash) in D:
        phis = np.array([p(s, a) for p in phi])
        phis_dash = np.array([p(s_dash, pi(s_dash, w)) for p in phi])
        A_tilde += phis*(phis-gamma*phis_dash).T
        b_tilde += phis*r
    return np.linalg.inv(A_tilde)*b_tilde


def lspi(D, phi, gamma, epsilon, w_0):
    w_dash = w_0
    while True:
        w = w_dash
        w_dash = lstdq(D, phi, gamma, w)
        if np.linalg.norm(x=(w-w_dash), ord=2) < epsilon:
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


def evaluate(policy, episodes):
    cumulative_rewards = []
    for e in range(episodes):
        state = env.reset()
        cumulative_reward = []
        while True:
            action = policy(state)
            next_state, reward, done, info = env.step(action)
            cumulative_reward.append(reward)
            if done:
                break
            state = np.copy(next_state)
        cumulative_rewards.append(cumulative_reward)
    # Average cumulative rewards over episodes
    cumulative_rewards = np.average(cumulative_rewards, axis=1)
    plt.figure()
    plt.plot(cumulative_rewards, label='cumulative reward averaged over episodes')
    plt.legend()
    plt.show()


def main():
    D = sample(epochs=10000)
    w_0 = [1, 1]
    gamma = 0.5
    epsilon = 1e-2
    w_star = lspi(D, kleinvieh, epsilon, gamma, w_0)
    policy_star = w_star * kleinvieh
    evaluate(policy_star, episodes=100)


if __name__ == '__main__':
    main()
