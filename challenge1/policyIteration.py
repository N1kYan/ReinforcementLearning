from __future__ import print_function
import sys
import numpy as np
import gym
import quanser_robots
from Regression import Regressor
from Evaluation import *
from DiscreteEnvironment import DiscreteEnvironment
from Utils import *

# TODO: comments...

def __policy_evaluation(env, policy, theta, gamma):
    value_function = np.zeros(env.state_space_size)
    while True:
        delta = 0
        for s in range(env.state_space_size):
            expected_reward = 0
            for a, prob_a in enumerate(policy[s]):
                for prob_s, next_state, reward, _ in env.P[s][a]:
                    expected_reward += prob_a * prob_s * (reward + gamma * value_function[next_state])
            delta = max(delta, np.abs(expected_reward - value_function[s]))
            value_function[s] = expected_reward
        if delta < theta:
            break
    return np.array(value_function)


def __policy_improvement(env, value_function, policy, gamma):
    for s in range(env.state_space_size):
        Q_sa = np.zeros(env.action_space_size)
        for a in range(env.action_space_size):
            for prob_s, next_state, reward, _ in env.P[s][a]:
                Q_sa[a] += prob_s * (reward + gamma * value_function[next_state])
        best_action = np.argmax(Q_sa)
        policy[s] = np.eye(env.action_space_size)[best_action]
    return policy


def policy_iteration(env, epochs, theta, gamma):
    # Initialize random policy
    policy = np.ones([env.state_space_size, env.action_space_size]) / env.action_space_size
    for i in range(epochs):
        value_function = __policy_evaluation(env, policy=policy, gamma=gamma, theta=theta)
        old_policy = np.copy(policy)
        new_policy = __policy_improvement(env, value_function, old_policy, gamma=gamma)
        if np.all(policy == new_policy):
            print ("policy stable.")
            break
        policy = np.copy(new_policy)
    return value_function, policy


def main():
    env = gym.make('Pendulum-v2')
    reg = Regressor()
    # Please use tuples for state and action space sizes
    #disc_env = DiscreteEnvironment(env=env, name='LowerBorder', state_space_size=(32+1, 16+1),
    #                              action_space_size=(16+1,))
    disc_env = DiscreteEnvironment(env=env, name='EasyPendulum', state_space_size=(16+1, 16+1),
                                   action_space_size=(16+1,))
    #regressorState, regressorReward = reg.perform_regression(epochs=10000, env=env, save_flag=False)
    value_function, policy = policy_iteration(env=disc_env, epochs=100, theta=1e-1, gamma=0.6)
    evaluate(env=env, episodes=100, disc = disc_env, policy=policy, render=False)
    visualize(policy=policy, value_function=value_function, state_space=disc_env.state_space)

if __name__ == "__main__":
    main()