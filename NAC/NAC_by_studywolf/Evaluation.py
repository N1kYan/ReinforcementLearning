import gym
import time
import numpy as np
import random
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import os
import datetime
from NAC_by_studywolf.actor_network import Actor


def evaluate(env, sess, actor, episodes=25):
    """
    Evaluates the learned agent on a given instance of MyEnvironment class.
    Evaluation is done by calculating cumulative reward of each episode and
    the average reward in each episode. We plot both and save it.

    :param env: MyEnvironment instance; The environment the agent is evaluated on
    :param sess: Tensorflow session
    :param actor: The actor (policy network) used for generating actions
    :param episodes: Number of episodes used for evaluation
    :return: None
    """

    time_steps = 10000

    print("\nEVALUATION: {} episodes with {} time steps each (or until 'done')"
          .format(episodes, time_steps))

    cumulative_episode_reward = []
    average_episode_reward = []
    trajectory_lengths = []

    for e in range(episodes):

        done = False
        observation = env.reset()

        undiscounted_return = 0
        rewards = []

        for t in range(time_steps):

            if done:
                trajectory_lengths.append(t)
                cumulative_episode_reward.append(undiscounted_return)
                average_episode_reward.append(np.mean(rewards))
                break

            action = actor.get_action(sess, observation)
            observation, reward, done, _ = env.step(action)

            undiscounted_return += reward
            rewards.append(reward)

    print("Average trajectory length:", np.mean(trajectory_lengths))
    print("Average episode reward:", np.mean(average_episode_reward))
    print("Average cumulative episode reward:",
          np.mean(cumulative_episode_reward))

    plot_save_rewards(env, cumulative_episode_reward, average_episode_reward)


def plot_save_rewards(env, cumulative_episode_reward, average_episode_reward):
    """
    Plot and save results of evaluation.

    :param env: Instance of MyEnvironment class; used to name the folder for saving plots
    :param cumulative_episode_reward: 2D-array, containing all episodes and the cumulative reward per step per episode
    :param average_episode_reward: 2D-array, episode x average reward per step per episode
    :return: None
    """

    # Plot & save average reward per episode
    plt.figure()
    plt.title("Average reward per episode")
    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    plt.plot(average_episode_reward)
    plt.savefig("{}/avg_reward.png".format(env.save_folder))
    plt.close()

    # Plot & save cumulative reward per episode
    plt.figure()
    plt.title("Cumulative reward per episode")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.plot(cumulative_episode_reward)
    plt.savefig("{}/cum_reward.png".format(env.save_folder))
    plt.close()


def render(env, sess, actor, episodes=10):
    """
    Renders the learned agent on a given instance of MyEnvironment class.

    :param env: MyEnvironment instance, on which the agent is evaluated on
    :param sess: Tensorflow session
    :param actor: The actor (policy network) used for generating actions
    :param episodes: Number of episodes used for rendering
    :return: None
    """

    time_steps = 10000

    print("\nRENDER: {} episodes with {} time steps each (or until 'done')"
          .format(episodes, time_steps))

    for e in range(episodes):

        print("Episode {} ... ".format(e), end='')
        sys.stdout.flush()

        done = False
        observation = env.reset()

        for t in range(time_steps):

            # Render environment
            env.render()
            time.sleep(0.1)

            # Break loop, if episode has finished
            if done:
                print("Episode ended after {} time steps!".format(t))
                break

            action = actor.get_action(sess, observation)
            observation, _, done, _ = env.step(action)
