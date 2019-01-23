from __future__ import print_function
import sys
import gym
import pickle
import quanser_robots
import numpy as np
import matplotlib.pyplot as plt


def basis_functions():
    """
    Defines the basis functions used for linear approximation.

    :return: The basis functions
    """
    kleinvieh = [
        lambda s, a: 1,
        lambda s, a: np.abs(np.sin(s[0])) * a,
        lambda s, a: np.abs(np.sin(s[1])) * a,
        lambda s, a: np.abs(np.sin(s[2])) * a,
        lambda s, a: np.abs(np.sin(s[3])) * a,
        lambda s, a: np.abs(np.sin(s[4])) * a,
        lambda s, a: np.abs(np.cos(s[0])) * a,
        lambda s, a: np.abs(np.cos(s[1])) * a,
        lambda s, a: np.abs(np.cos(s[2])) * a,
        lambda s, a: np.abs(np.cos(s[3])) * a,
        lambda s, a: np.abs(np.cos(s[4])) * a,
        lambda s, a: s[3] * a,
        lambda s, a: s[4] * a,

        # lambda s, a: np.exp(-np.linalg.norm(np.array([np.arctan(s[0]/s[1]), np.arctan(s[2]/s[3])] -
        #                                              np.array([-np.pi / 4, -1])) / 2)) * a,
        # lambda s, a: np.exp(-np.linalg.norm(np.array([np.arctan(s[0] / s[1]), np.arctan(s[2] / s[3])] -
        #                                              np.array([0, -1])) / 2)) * a,
        # lambda s, a: np.exp(-np.linalg.norm(np.array([np.arctan(s[0] / s[1]), np.arctan(s[2] / s[3])] -
        #                                              np.array([np.pi / 4, -1])) / 2)) * a,
        # lambda s, a: np.exp(-np.linalg.norm(np.array([np.arctan(s[0] / s[1]), np.arctan(s[2] / s[3])] -
        #                                              np.array([-np.pi / 4, 0])) / 2)) * a,
        # lambda s, a: np.exp(-np.linalg.norm(np.array([np.arctan(s[0] / s[1]), np.arctan(s[2] / s[3])] -
        #                                              np.array([0, 0])) / 2)) * a,
        # lambda s, a: np.exp(-np.linalg.norm(np.array([np.arctan(s[0] / s[1]), np.arctan(s[2] / s[3])] -
        #                                              np.array([np.pi / 4, 0])) / 2)) * a,
        # lambda s, a: np.exp(-np.linalg.norm(np.array([np.arctan(s[0] / s[1]), np.arctan(s[2] / s[3])] -
        #                                              np.array([-np.pi / 4, 1])) / 2)) * a,
        # lambda s, a: np.exp(-np.linalg.norm(np.array([np.arctan(s[0] / s[1]), np.arctan(s[2] / s[3])] -
        #                                              np.array([0, 1])) / 2)) * a,
        # lambda s, a: np.exp(-np.linalg.norm(np.array([np.arctan(s[0] / s[1]), np.arctan(s[2] / s[3])] -
        #                                              np.array([np.pi / 4, 1])) / 2)) * a,

    ]
    return kleinvieh


def pi(s, w,):
    """
    Policy returning the liner combination of basis functions and weights
    :param s: state
    :param w: weights
    :return: Linear combination (action)
    """
    def policy_a(a):
        su = 0
        for (mist, w_dash) in zip(basis_functions(), w):
            dash = mist(s, a) * w_dash
            su += dash
        return su
    global A
    return np.array([A[np.argmax([policy_a(a) for a in A])]])


def lstdq_opt(D, phi, gamma, w):
    """
    Optimized LSTDQ
    :param D: Samples
    :param phi: Basis functions
    :param gamma: Discount factor
    :param w: Current weights
    :return: TODO
    """
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


def lstdq(D, phi, gamma, w):
    """
    Least-Squares Temporal-Difference Q-Learning
    :param D: Samples
    :param phi: Basis functions
    :param gamma: Discount factor
    :param w: Current weights
    :return: TODO
    """
    k = len(phi)
    A_tilde = np.zeros(shape=(k, k))
    b_tilde = np.zeros(shape=(k, 1))
    for (s, a, r, s_dash) in D:
        phis = np.array([p(s, a) for p in phi]).reshape(-1, 1)
        phis_dash = np.array([p(s_dash, pi(s_dash, w)) for p in phi])
        A_tilde = A_tilde + phis*(phis-gamma*phis_dash).T
        b_tilde = b_tilde + phis*r
    return np.matmul(np.linalg.inv(A_tilde), b_tilde)


def lspi(D, phi, gamma, epsilon, w_0):
    """
    Least-Squares Policy Iteration
    :param D: Samples
    :param phi: Basis functions
    :param gamma: Discount factor
    :param epsilon: Convergence criterion
    :param w_0: Initial weights
    :return: Learned optimal weights
    """
    w_dash = w_0
    while True:
        w = w_dash
        w_dash = lstdq(D=D, phi=phi, gamma=gamma, w=w)
        # w_dash = lstdq_opt(D=D, phi=phi, gamma=gamma, w=w)
        diff = np.linalg.norm(x=(w-w_dash), ord=2)
        print("Diff {} < {}: {} ".format(diff, epsilon, diff < epsilon))
        if diff < epsilon:
            break
    return w_dash  # TODO: w?


def sample(epochs):
    """
    Samples episodes from the environment.
    :param epochs: Number of episodes for sampling
    :return: Sample Matrix
    """
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


def evaluate(w_star=None, episodes=25, render=False, plot=True, load_flag=True):
    """
    Evaluate the learned weights/policy.
    :param w_star: Learned weights
    :param episodes: Episodes for evaluation
    :param render: Set true to render the episodes
    :param plot: Set true to plot rewards per episode
    :param load_flag: Set true to lead learned weights from file
    :return: None
    """
    if load_flag:
        w_star = pickle.load(open("lspi_weights.p", "rb"))

    print("Evaluating...", end='')
    sys.stdout.flush()
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
    if plot:
        plt.show()
    print("done")


def train(my_env, sample_epochs=10000, gamma=0.99, epsilon=1e-4, save_flag=True):
    """
    Set environment, discretize action Space and run LSPI training.
    :param my_env: The gym environment
    :param sample_epochs: Episodes for samling D
    :param gamma: Discount factor gamma  # 0.95
    :param epsilon: Convergence criterium  # 1
    :param save_flag: Set true to save learned weights to file
    :return: learned weights w_star
    """
    # Define gym environment
    global env
    env = gym.make(my_env)
    print(env.spec)
    print("State Space Low:", env.observation_space.low)
    print("State Space Hight:", env.observation_space.high)

    # Set discrete action space
    global A
    A = np.linspace(start=env.action_space.low, stop=env.action_space.high, num=7)  # 13, 25, 49
    print("Discrete Action Space: ", A)

    # Sample from environment
    print("Sampling...", end='')
    sys.stdout.flush()
    D = sample(epochs=sample_epochs)
    print("done")

    # Initialize weights
    w_0 = np.ones(shape=(len(basis_functions()), 1))
    for i in range(len(w_0)):
        w_0[i] = np.random.uniform(low=-1, high=1)

    # Run LSPI algorithm
    print("Learning...")
    w_star = lspi(D=D, phi=basis_functions(), gamma=gamma, epsilon=epsilon, w_0=w_0)
    print("w*:", w_star)
    print("...done")

    # Save learned weights in text file
    if save_flag:
        pickle.dump(w_star, open("lspi_weights.p", "wb"))

    # Return learned weights
    return w_star


def main():
    w_star = train(my_env='CartpoleStabShort-v0')
    evaluate(w_star=w_star)
    # evaluate()


if __name__ == '__main__':
    main()
