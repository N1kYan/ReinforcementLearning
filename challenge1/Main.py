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
        gamma = 0.01

        # Perform dynamic programming to get value function and near optimal policy
        print("Training episode 0...")
        disc_env.update_transition_probabilities(policy=None, epochs=10000)
        value_function, policy = \
            value_iteration(regressorState=regressorState,
                            regressorReward=regressorReward, disc=disc_env, vf=None, theta=theta, gamma=gamma,
                            use_true_model=use_true_model_flag)
        evaluate(env=env, episodes=100, disc=disc_env, policy=policy, render=False)
        visualize(value_function, policy, disc=disc_env)
        """
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


"""
    Evaluation stuff to see the predictions, discretizations and learned functions in action
"""
def evaluate(env, episodes, disc, policy, render):

    rewards_per_episode = []

    print("Evaluating...")

    for e in range(episodes):

        # Discretize first state
        state = env.reset()
        index = disc.map_to_state(state)

        cumulative_reward = [0]

        for t in range(200):
            # Render environment
            if render:
                env.render()
            #time.sleep(2)

            # Do step according to policy and get observation and reward
            action = np.array([policy[index[0], index[1]]])

            state, reward, done, info = env.step(action)

            cumulative_reward.append(cumulative_reward[-1] + reward)

            # Discretize observed state
            index = disc.map_to_state(state)

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



def visualize(value_function, policy, disc=None):
    plt.figure()
    plt.title("Value function")
    plt.imshow(value_function)
    plt.colorbar()

    if disc is not None:
        plt.ylabel("Angle in Radians")
        plt.yticks(range(disc.state_space_size[0]), labels=disc.state_space[0].round(2))
        plt.xlabel("Velocity")
        plt.xticks(range(disc.state_space_size[1]), labels=disc.state_space[1].round(1))

    plt.show()

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
    disc_env = DiscreteEnvironment(env, 'EasyPendulum', state_space_size=(16 + 1, 16 + 1),
                                   action_space_size=(16+1,))

    #print("-------------------------\nState Space Discretization:")
    #print(disc_env.state_space)
    #print("-------------------------\nAction Space Discretization:")
    #print(disc_env.action_space)

    value_function, policy = training(env=env, disc_env=disc_env, value_function=None, load_regression_flag=True,
                                       load_value_function_flag=False, use_true_model_flag=False)

    evaluate(env=env, episodes=100, disc=disc_env, policy=policy, render=True)
    visualize(value_function, policy)


if __name__ == "__main__":
    main()
