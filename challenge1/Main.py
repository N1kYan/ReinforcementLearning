from __future__ import print_function
import gym
import quanser_robots
import time
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from Discretization import PendulumDiscretization
from Discretization import EasyPendulumDiscretization
from Regression import Regressor
from DynamicProgramming import value_iteration
from DynamicProgramming import policy_iteration
from Utils import *
import TrueModel
"""
    Training method
    
    Performs regression on the environment to learn state-transition- and reward function.
    
    Learns optimal policy with dynamic programming methods afterwards.
    
"""


def training(env, regression_flag, true_model_flag):
    """
    Learning the value and policy function for our environment.
    :param env: The environment we want to learn the value and policy function
    of.
    :param regression_flag: Set to false if we want to use the value & policy
    function saved in "training.pkl" instead of estimating them via regression.
    :param true_model_flag: Set to true, if we want to use the underlying true
    model instead of estimating the value and policy function via Regression.
    :return: the value function object, the policy function object and the
    discretization object which we used to create the value/policy function
    """
    # DISCRETIZATION
    disc = PendulumDiscretization(state_space_size=(16 + 1, 16 + 1), action_space_size=16 + 1)

    print("-------------------------\nState Space Discretization:")
    print(disc.state_space)
    print("-------------------------\nAction Space Discretization:")
    print(disc.action_space)

    regressorState = None
    regressorReward = None

    # REGRESSION
    # We want to use regression to learn the rewards and state transitions
    if not true_model_flag:
        reg = Regressor()

        # Learning episodes / amount of samples for regression
        epochs = 10000

        # Perform regression
        regressorState, regressorReward = reg.perform_regression(epochs, env, regression_flag)

    # Perform dynamic programming to get value function and near optimal policy
    value_function, policy = \
        value_iteration(regressorState=regressorState,
                        regressorReward=regressorReward, disc=disc,
                        theta=0.01, gamma=0.7,
                        use_true_model=true_model_flag)
    print("-------------------------\nValue Function:")
    print(value_function)
    print("-------------------------\nPolicy Function:")
    print(policy)
    return value_function, policy, disc



"""
    Evaluation stuff to see the predictions, discretizations and learned functions in action

"""
def evaluate(env, disc, policy, render):

    rewards_per_episode = []

    episodes = 100

    print("Evaluating...")

    for e in range(episodes):

        # Discretize first state
        state = env.reset()
        index = disc.map_to_index(state)

        cumulative_reward = [0]

        for t in range(200):
            # Render environment
            if render:
                env.render()

            # Do step according to policy and get observation and reward
            action = np.array([policy[index[0], index[1]]])

            state, reward, done, info = env.step(action)

            cumulative_reward.append(cumulative_reward[-1] + reward)

            # Discretize observed state
            index = disc.map_to_index(state)

            if done:
                print("Episode {} finished after {} timesteps".format(e + 1, t + 1))
                break

        rewards_per_episode.append(cumulative_reward)

    print("...done")

    # TODO: Look at calculation of mean cumulative rewards
    # Average reward over episodes
    rewards = np.average(rewards_per_episode, axis=0)

    env.close()

    # Plot rewards per timestep averaged over episodes
    plt.figure()
    plt.plot(rewards, label='Cumulative reward per timestep, averaged over {} episodes'.format(episodes))
    plt.legend()
    plt.show()



def visualize(value_function, policy):
    print()
    dim = np.shape(value_function)
    for a in np.arange(dim[0]):
        max = 0
        print("Value for {},x is".format(a),end='')
        for b in np.arange(dim[1]):
            print(" ",value_function[a, b], end='')
            """
            if value_function[a, b] > max:
                max = value_function[a, b]
                ind = b
            """
        print (".")

    print()
    dim = np.shape(policy)
    for a in np.arange(dim[0]):
        max = 0
        print("Policy for {},x is".format(a), end='')
        for b in np.arange(dim[1]):
            print(" ", policy[a, b], end='')
            """
            if value_function[a, b] > max:
                max = value_function[a, b]
                ind = b
            """
        print(".")



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

    print("State space:  Shape:{}  Min:{}  Max:{} "
          .format(np.shape(env.observation_space), env.observation_space.low,
                  env.observation_space.high))
    print("Action space:  Shape:{}  Min:{}  Max:{} "
          .format(np.shape(env.action_space), env.action_space.low,
                  env.action_space.high))

    # Search for value function and regression files,
    # if none exists, perform learning and evaluation and save value function and regression files
    regression_flag = True  # Set to False to load regressors from file
    true_model_flag = True  # Set to True to perform dp with true model instead of model gotten by regression
    value_function_save_flag = True  # Set to False to load value function from file
    # Value function visualisation is only done when set to False

    if open('vf.pkl') and not value_function_save_flag:
        print()
        print("Found value function file.")
        with open('vf.pkl', 'rb') as pickle_file:
            (vf, policy) = pickle.load(pickle_file)
        visualize(vf, policy)
    else:
        value_function, policy, disc \
            = training(env=env, regression_flag=regression_flag,
                       true_model_flag=true_model_flag)
        save_object((value_function, policy), 'vf.pkl')
        evaluate(env=env, disc=disc, policy=policy, render=True)
        # visualize(value_function, policy)


if __name__ == "__main__":
    main()