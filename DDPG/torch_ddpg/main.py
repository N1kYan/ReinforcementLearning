import gym
import quanser_robots
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time
import sys
from torch_ddpg.DDPGAgent import Agent
from torch_ddpg.ActionNoise import OUNoise


# TODO: Noise Source
# TODO: Plot von Bell_error & Regression_error (Yannik Frisch)
# TODO: Replay Buffer füllen lassen
# TODO: Alle Parameter werden in der Main gesetzt
# TODO: Batch normalization (zum Schluss)
# TODO: Importance Sampling (Alex)
# TODO: Forumeintrag anschauen und einarbeiten


BUFFER_SIZE = int(1e6)  # replay buffer size #1e6
BATCH_SIZE = 1024      # minibatch size #64
GAMMA = 0.9            # discount factor #0.99
TAU = 1e-2                # for soft update of target parameters #1e-3
LR_ACTOR = 1e-4        # learning rate of the actor #1e-4
LR_CRITIC = 1e-3        # learning rate of the critic #1e-3
WEIGHT_DECAY = 0        # L2 weight decay #1e-2


#env = gym.make('Qube-v0')
#env = gym.make('Pendulum-v0')
env = gym.make('BallBalancerSim-v0')
print(env.spec)
print("State Space Shape: {}\nLow: {}\nHigh: {}".format(np.shape(env.reset()),
                                                        env.observation_space.low,
                                                        env.observation_space.high))
print("Action Space Shape: {}\nLow: {}\nHigh: {}".format(np.shape(env.action_space.sample()),
                                                         env.action_space.low,
                                                         env.action_space.high))
env_observation_size = len(env.reset())
env_action_size = len(env.action_space.sample())
env_action_low = env.action_space.low
env_action_high = env.action_space.high
random_seed = 3
update_frequency = 1

env.seed(random_seed)

# Noise generating process
OU_NOISE = OUNoise(size=env_action_size, seed=random_seed, mu=0., theta=0.15, sigma=0.2)


# DDPG learning agent
AGENT = Agent(state_size=env_observation_size, action_size=env_action_size,
              action_bounds=(env_action_low, env_action_high), random_seed=random_seed, buffer_size=BUFFER_SIZE,
              batch_size=BATCH_SIZE, gamma=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC,
              weight_decay=WEIGHT_DECAY, noise_generator=OU_NOISE)


def evaluation(epochs=25, render=False):
    """
    Evaluates the trained agent on the environment (both declaired above).
    Plots the reward per timestep for every episode.
    :param epochs: Episodes for evaluation
    :param render: Set true to render the evaluation episodes
    :return: None
    """
    plt.figure()
    plt.title("Rewards during evaluation")
    plt.xlabel("Time-step")
    plt.ylabel("Current reward")
    for e in range(1, epochs + 1):
        state = env.reset()
        rewards = []
        t = 0
        while True:
            t += 1
            if render:
                env.render()
            action = AGENT.act(state)
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            state = np.copy(next_state)
            if done:
                break
            plt.plot(rewards)
        env.close()


def training(epochs=2000, max_steps=500, epoch_checkpoint=100, render=True):
    """
    Runs the training process on the gym environment.
    Then plots the cumulative reward per episode.
    :param epochs: Number of epochs for training
    :param max_steps: Maximum time-steps for each training epoch;
     Does end epochs for environments, which epochs are not time limited
    :param epoch_checkpoint: Checkpoint for printing the learning progress and rendering the environment
    :param render: Set true for rendering every 'epoch_checkpoint' episode
    :return: None
    """

    # Reset agent's noise generator
    AGENT.reset()

    # Measure the time we need to learn
    time_start = time.time()

    critic_loss = []

    scores_deque = deque(maxlen=epoch_checkpoint)
    e_cumulative_rewards = []
    for e in range(1, epochs + 1):
        state = env.reset()
        # AGENT.reset()
        cumulative_reward = 0
        t = 0
        for t_i in range(max_steps):
            t += 1
            if (e % epoch_checkpoint == 0) and render:
                env.render()
            action = AGENT.act(state)
            # print(action)
            next_state, reward, done, _ = env.step(action)
            # print(reward)
            # reward = reward*1000  # Qube-v0 rewards are VERY small; only for debugging
            AGENT.step(state, action, reward, next_state, done)
            state = next_state
            cumulative_reward += reward
            if done:
                break
        env.close()
        scores_deque.append(cumulative_reward)
        e_cumulative_rewards.append(cumulative_reward)
        print('\rEpisode {}\tAverage Reward: {}\tSteps: {}\t({:.2f} min elapsed)'.
              format(e, np.mean(scores_deque), t, (time.time() - time_start)/60), end="")
        if e % epoch_checkpoint == 0:
            # Print cumulative reward per episode averaged over #epoch_checkpoint episodes
            print('\rEpisode {}\tAverage Reward: {:.3f}\t({:.2f} min elapsed)'.
                  format(e, np.mean(scores_deque), (time.time() - time_start)/60))

    print("Learning weights took {:.2f} min.".format((time.time() - time_start) / 60 ))
    print("Final average cumulative reward", np.mean(e_cumulative_rewards))

    # Plot the cumulative reward per episode during training process
    fig = plt.figure()
    plt.title("Cumulative reward during training")
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(e_cumulative_rewards) + 1), e_cumulative_rewards)
    plt.ylabel('Cumulative reward')
    plt.xlabel('Episode #')


if __name__ == "__main__":
    training()
    evaluation()
    plt.show()

