import numpy as np
import gym
import quanser_robots
from Regression import Regressor
from Evaluation import *
from DiscreteEnvironment import DiscreteEnvironment

def policy_iteration(env, disc_env, regressorState, regressorReward, theta, gamma):

    # Initialize value function and policy
    value_function = np.zeros(shape=disc_env.state_space_size)
    policy = np.zeros(shape=disc_env.state_space_size)

    def policy_evaluation():
        while True:
            delta = 0
            # TODO: Bellman in matrix equation?
            for s0 in disc_env.state_space[0]:
                for s1 in disc_env.state_space[1]:
                    index = disc_env.map_to_state([s0, s1])
                    v =  value_function[index[0], index[1]]
                    a = disc_env.map_to_action([policy[index[0], index[1]]])
                    regression_input = np.array([s0, s1, a]).reshape(1, -1)
                    print(regression_input)

                    expected_reward = regressorReward.predict(regression_input) +\
                                      gamma * regressorState.predict(regression_input)
                    print(expected_reward)
                    value_function[index[0], index[1]] = expected_reward
                    delta = max(delta, np.abs(v-expected_reward))
            print("Delta: ", delta)
            if delta < theta:
                break
        policy_improvement()

    def policy_improvement():
        policy_stable = True
        while True:
            for s0 in disc_env.state_space[0]:
                for s1 in disc_env.state_space[1]:
                    index = disc_env.map_to_state([s0, s1])
                    old_action = policy[index[0], index[1]]

                    max_reward = -10
                    max_action = None
                    for a in disc_env.action_space[0]:
                        regression_input = [s0, s1, disc_env.map_to_action([a])]
                        expected_reward = regressorReward.predict(regression_input) + \
                                          gamma * regressorState.predict(regression_input)
                        if expected_reward >= max_reward:
                            max_action = a
                    policy[index[0], index[1]] = max_action
                    if old_action != max_action:
                        policy_stable = False
            if policy_stable:
                break
            else:
                policy_evaluation()

    policy_evaluation()
    return policy


def main():
    env = gym.make('Pendulum-v2')
    reg = Regressor()
    # Please use tuples for state and action space sizes
    disc_env = DiscreteEnvironment(env=env, name='EasyPendulum', state_space_size=(16+1, 16+1),
                                   action_space_size=(8+1,))
    regressorState, regressorReward = reg.perform_regression(epochs=1000, env=env)
    policy = policy_iteration(env, disc_env, regressorState, regressorReward, theta=1e-2, gamma=0.1)
    evaluate(env=env, episodes=100 , disc = disc_env, policy=policy, render=True)

if __name__ == "__main__":
    main()