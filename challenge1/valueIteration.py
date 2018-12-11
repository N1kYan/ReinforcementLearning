from __future__ import print_function
import numpy as np
import gym
import pickle
import quanser_robots
from DiscreteEnvironment import DiscreteEnvironment
from Evaluation import *
from Utils import *

# TODO: comments


def value_iteration(env, theta, gamma, save_flag):

    if not save_flag and open('./pickle/value_iteration.pkl'):
        print()
        print("Found value iteration file.")
        print()
        with open('./pickle/value_iteration.pkl', 'rb') as pickle_file:
            (vf, p) = pickle.load(pickle_file)
        return vf, p

    else:
        print("Value Iteration ... ")
        # Initialize value function
        value_function = np.zeros(env.state_space_shape)

        # Iterate to converge to optimal value function
        while True:
            delta = 0
            for s0 in env.state_space[0]:
                for s1 in env.state_space[1]:
                    index = env.map_to_state([s0, s1])
                    v = value_function[index[0], index[1]]
                    max_reward = None
                    for a in env.action_space[0]:
                        # Sum up over all possible successors
                        successors = env.get_successors(state=[s0, s1], action=[a])
                        expected_reward = None
                        for item in successors.items():
                            prob = item[1][0]
                            reward = item[1][1]
                            succ_index = [item[0][0], item[0][1]]
                            if expected_reward is None:
                                expected_reward = prob*(reward + gamma*value_function[succ_index[0], succ_index[1]])
                            else:
                                expected_reward += prob*(reward + gamma*value_function[succ_index[0], succ_index[1]])
                        if (max_reward is None) or expected_reward>max_reward:
                            max_reward = expected_reward
                    value_function[index[0], index[1]] = max_reward
                    delta = max(delta, np.abs(v - max_reward))
            print ("Delta =", delta, end='')
            if delta < theta:
                print(" < Theta =", theta)
                break
            else:
                print(" > Theta =", theta)

        # Initialize policy
        policy = np.zeros(env.state_space_shape)

        print("Defining policy ... ")
        # Iterate to converge to optimal policy
        for s0 in env.state_space[0]:
            for s1 in env.state_space[1]:
                max_reward = None
                best_action = None
                for a in env.action_space[0]:
                    successors = env.get_successors(state=[s0, s1], action=[a])
                    expected_reward = None
                    for item in successors.items():
                        prob = item[1][0]
                        reward = item[1][1]
                        succ_index = [item[0][0], item[0][1]]
                        if expected_reward is None:
                            expected_reward = prob * (reward + gamma * value_function[succ_index[0], succ_index[1]])
                        else:
                            expected_reward += prob * (reward + gamma * value_function[succ_index[0], succ_index[1]])
                    if (max_reward is None) or expected_reward>max_reward:
                        max_reward = expected_reward
                        best_action = a
                index = env.map_to_state([s0, s1])
                policy[index[0], index[1]] = best_action
        print("... done!")
        print()
        print("Saving value iteration file.")
        print()
        save_object((value_function, policy), './pickle/value_iteration.pkl')

        return value_function, policy


def main():
    env = gym.make('Pendulum-v2')
    disc_env = DiscreteEnvironment(env=env, name='EasyPendulum',
                                   state_space_shape=(4+1, 8+1),
                                   action_space_shape=(32+1,), gaussian_granularity=1, gaussian_sigmas=[0.3, 0.3])
    state = env.reset()
    disc_env.perform_regression(env=env, epochs=10000, save_flag=False)

    # Covariance matrix instead of sigma list??
    # disc_env.get_successors(state=state, action=[1.0], sigmas=[.1, .1])
    value_function, policy = value_iteration(env=disc_env, theta=1e-1, gamma=0.5, save_flag=False)
    evaluate(disc_env=disc_env, episodes=100, policy=policy, render=False, sleep=0)
    visualize(value_function, policy, state_space=disc_env.state_space)



if __name__ == "__main__":
    main()