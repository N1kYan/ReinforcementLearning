from __future__ import print_function
import gym
import quanser_robots
import time
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from Regression import Regressor
from DynamicProgramming import value_iteration
from DynamicProgramming import policy_iteration
from DiscreteEnvironment import DiscreteEnvironment
from Utils import *


"""
    Training method
    
    Performs regression on the environment to learn state-transition- and reward function.
    
    Learns optimal policy with dynamic programming methods afterwards.    
"""
def training(env, disc_env, value_function, load_regression_flag, load_value_function_flag, use_true_model_flag):

    """
    Learning the value and policy function for our environment.
    :param env: The environment we want to learn the value and policy function
    of.
    :param regression_flag: Set to false if we want to use the value & policy
    function saved in "training.pkl" instead of estimating them via regression.
    :param true_model_flag: Set to true, if we want to use the underlying true
    model instead of estimating the value and policy function via Regression.
    :return: the value function object, the policy function object
    """

    regressorState = None
    regressorReward = None

    # TODO: Regression should be done with the same samples as the transition probability estimate
    # Load saved regressors when any are found and loading flag set True
    if load_regression_flag and open('./pickle/reg.pkl'):
        print()
        print("Found regression file.")
        print()
        with open('./pickle/reg.pkl', 'rb') as pickle_file:
            (rS, rR) = pickle.load(pickle_file)
        regressorState = rS
        regressorReward = rR
    else:
        reg = Regressor()
        # Learning episodes / amount of samples for regression
        epochs = 10000
        # Perform regression
        regressorState, regressorReward = reg.perform_regression(epochs, env)

    # Perform dynamic programming to get value function and near optimal policy
    value_function = None
    policy = None

    # Load saved value function when any is found and loading flag set True
    if load_value_function_flag and open('./pickle/vf.pkl'):
        print()
        print("Found value function file.")
        print()
        with open('./pickle/vf.pkl', 'rb') as pickle_file:
            (vf, p) = pickle.load(pickle_file)
            value_function = vf
            policy = p
    else:
        theta = 1e-1
        gamma = 0.95

        # Perform dynamic programming to get value function and near optimal policy


        disc_env.update_transition_probabilities(policy=None, epochs=10000)
        value_function, policy = policy_iteration(regressorState, regressorReward,
                                                  disc = disc_env, theta=theta, gamma=gamma)
        """
        print("Training episode 0...")
        disc_env.update_transition_probabilities(policy=None, epochs=10000)
        value_function, policy = \
            value_iteration(regressorState=regressorState,
                            regressorReward=regressorReward, disc=disc_env, vf=None, theta=theta, gamma=gamma,
                            use_true_model=use_true_model_flag)
        evaluate(env=env, episodes=100, disc=disc_env, policy=policy, render=False)
        visualize(value_function, policy, disc=disc_env)
        
        for i in range(19):
            print("Training episode {}...".format(i+1))
            disc_env.update_transition_probabilities(policy=policy, epochs=500)
            value_function, policy = \
                value_iteration(regressorState=regressorState,
                                regressorReward=regressorReward, disc=disc_env, vf=value_function, theta=theta,
                                gamma=gamma, use_true_model=use_true_model_flag)
            #evaluate(env, disc_env, policy, render=True)
        """

        save_object((value_function, policy), './pickle/vf.pkl')
        #print("-------------------------\nValue Function:")
        #print(value_function)
        #print("-------------------------\nPolicy Function:")
        #print(policy)

    return value_function, policy



# TODO: True Model benutzen
# TODO: Other Discretization
# TODO: Transition Probabilities
# TODO: Ausprobieren mit ganz vielen States


def main():
    """Run dynamic programming on the quanser pendulum."""

    """
        The Pendulum-v2 environment:

        The action space is a Box(1,) with values between [-2,2] (joint effort)

        The state space is the current angle in radians & the angular velocity
        [min:-pi,-8; max:pi,8].
    """
    env = gym.make('Pendulum-v2') # Create gym/quanser environment
    disc_env = DiscreteEnvironment(env, 'EasyPendulum', state_space_size=(40 + 1, 30 + 1),
                                   action_space_size=(16+1,))

    #print("-------------------------\nState Space Discretization:")
    #print(disc_env.state_space)
    #print("-------------------------\nAction Space Discretization:")
    #print(disc_env.action_space)

    value_function, policy = training(env=env, disc_env=disc_env, value_function=None, load_regression_flag=True,
                                       load_value_function_flag=False, use_true_model_flag=False)

    evaluate(env=env, episodes=100, disc=disc_env, policy=policy, render=False)
    visualize(value_function, policy)


if __name__ == "__main__":
    main()
