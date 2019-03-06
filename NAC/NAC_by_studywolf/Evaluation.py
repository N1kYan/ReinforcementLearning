import gym
import time
import numpy as np
import random
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import os
import datetime


def evaluate(env, sess, policy_grad, render=False, episodes=25):
    """
    Evaluates the learned agent on a given instance of MyEnvironment class.
    Then runs plot_save.
    :param env: MyEnvironment instance; The environment the agent is evaluated on
    :param sess: Tensorflow session
    :param policy_grad: TODO
    :param render: Set true to render each episode of evaluation
    :param episodes: Number of episodes used for evaluation
    :return: None
    """
    time_steps = 10000

    print("\nEVALUATION: {} episodes with {} time steps each (or until 'done')"
          .format(episodes, time_steps))

    # Unpack the policy network (generates control policy)
    (pl_state, pl_actions, pl_advantages,
        pl_calculated, pl_optimizer) = policy_grad

    cumulative_episode_reward = []
    average_episode_reward = []

    for e in range(episodes):
        print("Episode {} ... ".format(e), end='')
        sys.stdout.flush()
        done = False
        observation = env.reset()
        undiscounted_return = 0
        rewards = []
        for t in range(time_steps):
            # Render environment
            if render:
                env.render()
                time.sleep(0.1)

            if done:
                print("Episode ended after {} time steps!".format(t))
                cumulative_episode_reward.append(undiscounted_return)
                average_episode_reward.append(np.mean(rewards))
                break

            obs_vector = np.expand_dims(observation, axis=0)
            probs = sess.run(
                pl_calculated,
                feed_dict={pl_state: obs_vector})

            # Check which action to take
            # stochastically generate action using the policy output
            probs_sum = 0
            action_i = None
            rnd = random.uniform(0, 1)
            for k in range(len(env.action_space)):
                probs_sum += probs[0][k]
                if rnd < probs_sum:
                    action_i = k
                    break
                elif k == (len(env.action_space) - 1):
                    action_i = k
                    break

            # Get the action (not only the index)
            # and take the action in the environment
            # Try/Except: Some env need action in an array
            action = env.action_space[action_i]
            try:
                observation, reward, done, info = env.step(action)
            except AssertionError:
                action = np.array([action])
                observation, reward, done, info = env.step(action)

            undiscounted_return += reward
            rewards.append(reward)

    plot_save(env, cumulative_episode_reward, average_episode_reward, True)


def plot_save(env, cumulative_episode_reward, average_episode_reward,
              save_flag=True):
    """
    Plot or save results of evaluation.
    :param env: Instance of MyEnvironment class; used to name the folder for saving plots
    :param cumulative_episode_reward: 2D-array, containing all episodes and the cumulative reward per step per episode
    :param average_episode_reward: 2D-array, episode x average reward per step per episode
    :param save_flag: Set true to save plots instead of showing them instantly
    :return: None
    """

    # Plot/Save average reward per episode
    plt.figure()
    plt.title("Average reward per episode")
    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    plt.plot(average_episode_reward)
    if save_flag:
        plt.savefig("{}/avg_reward.png".format(env.save_folder))
        plt.close()

    # Plot/Save cumulative reward per episode
    plt.figure()
    plt.title("Cumulative reward per episode")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.plot(cumulative_episode_reward)

    if save_flag:
        plt.savefig("{}/cum_reward.png".format(env.save_folder))
        plt.close()
    else:
        plt.show()
