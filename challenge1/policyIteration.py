from __future__ import print_function
import sys
import numpy as np
import gym
import quanser_robots
from Regression import Regressor
from Evaluation import *
from DiscreteEnvironment import DiscreteEnvironment
from Utils import *

def policy_iteration(env, disc_env, regressorState, regressorReward, theta, gamma):

    # Initialize value function and policy
    value_function = np.zeros(shape=disc_env.state_space_size)
    policy = np.zeros(shape=disc_env.state_space_size)

    def policy_evaluation():
        print("Policy Evaluation ...")
        while True:
            delta = 0
            # TODO: Bellman in matrix equation?
            for s0 in disc_env.state_space[0]:
                for s1 in disc_env.state_space[1]:

                    index = disc_env.map_to_state([s0, s1])
                    #print("{}|{} -> {}".format(s0, s1, index))
                    v = value_function[index[0], index[1]]
                    a = disc_env.map_to_action([policy[index[0], index[1]]])

                    regression_input = np.array([s0, s1, a]).reshape(1, -1)
                    new_index = disc_env.map_to_state(regressorState.predict(regression_input)[0])
                    expected_reward = regressorReward.predict(regression_input) +\
                                      gamma * value_function[new_index[0], new_index[1]]

                    value_function[index[0], index[1]] = expected_reward
                    delta = max(delta, np.abs(v-expected_reward)[0])
            print("Delta: ", delta)
            if delta < theta:
                break
        print()
        return delta

    def policy_improvement(delta):
        #if delta <= 1e-6:
        #    print(" Exit")
        #    return True
        print("Policy Improvement ... ", end='')
        sys.stdout.flush()
        policy_stable = True
        for s0 in disc_env.state_space[0]:
            for s1 in disc_env.state_space[1]:
                index = disc_env.map_to_state([s0, s1])
                old_action = policy[index[0], index[1]]

                max_reward = -25
                max_actions = [old_action] #TODO: ???
                # TODO: What happens for actions with equal expected reward?
                for a in disc_env.action_space[0]:
                    regression_input = np.array([s0, s1, disc_env.map_to_action([a])]).reshape(1, -1)
                    new_index = disc_env.map_to_state(regressorState.predict(regression_input)[0])
                    expected_reward = regressorReward.predict(regression_input) + \
                                      gamma * value_function[new_index[0], new_index[1]]

                    if expected_reward > max_reward:
                        max_actions = []
                        max_actions.append(a)
                        max_reward = expected_reward
                    #elif expected_reward == max_reward:
                    #    max_actions.append(a)

                max_actions = np.array(max_actions)
                # Sample random for multiple actions with max expected rewards
                policy[index[0], index[1]] = np.random.choice(max_actions) #TODO: not used, always take first?
                #print("old action {} policy {}".format(old_action, policy[index[0], index[1]]))
                if old_action != policy[index[0], index[1]]:
                    policy_stable = False

        if policy_stable:
            print(" policy stable")
            print()
        else:
            print(" policy not stable")
            print()

        return policy_stable

    policy_stable = False
    while not policy_stable:
        delta = policy_evaluation()
        policy_stable = policy_improvement(delta)

    return value_function, policy


def main():
    env = gym.make('Pendulum-v2')
    reg = Regressor()
    # Please use tuples for state and action space sizes
    disc_env = DiscreteEnvironment(env=env, name='LowerBorder', state_space_size=(32+1, 16+1),
                                   action_space_size=(16+1,))

    regressorState, regressorReward = reg.perform_regression(epochs=10000, env=env, save_flag=False)
    value_function, policy = policy_iteration(env, disc_env, regressorState, regressorReward, theta=1e-1, gamma=0.6)
    evaluate(env=env, episodes=100, disc = disc_env, policy=policy, render=False)
    visualize(policy=policy, value_function=value_function, state_space=disc_env.state_space)

if __name__ == "__main__":
    main()