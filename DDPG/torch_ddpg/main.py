import gym
import quanser_robots
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time
import sys
from torch_ddpg.DDPGAgent import Agent


# TODO: Noise Source
# TODO: Plot von Bell_error & Regression_error (Yannik Frisch)
# TODO: Replay Buffer füllen lassen
# TODO: Alle Parameter werden in der Main gesetzt
# TODO: Batch normalization (zum Schluss)
# TODO: Importance Sampling (Alex)
# TODO: Forumeintrag anschauen und einarbeiten



BUFFER_SIZE = int(1e6)  # replay buffer size #100000
BATCH_SIZE = 1024       # minibatch size #128
GAMMA = 0.99            # discount factor #0.99
TAU = 0.01                 # for soft update of target parameters #1e-3
LR_ACTOR = 1e-5         # learning rate of the actor #1e-5
LR_CRITIC = 1e-4        # learning rate of the critic #1e-4
WEIGHT_DECAY = 0        # L2 weight decay #0




# env = gym.make('Qube-v0')
# env = gym.make('Pendulum-v0')
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

agent = Agent(state_size=env_observation_size, action_size=env_action_size,
              action_bounds=(env_action_low, env_action_high), random_seed=random_seed, buffer_size=BUFFER_SIZE,
              batch_size=BATCH_SIZE, gamma=GAMMA, tau=TAU, LR_Actor=LR_ACTOR, LR_Critic=LR_CRITIC, weight_decay=WEIGHT_DECAY)


def training(epochs=2000, max_steps=500, epoch_checkpoint=250):
    """
    Runs the training process on the gym environment.
    :param epochs: Number of epochs for training
    :param max_steps: Maximum time-steps for each training epoch;
     Does end epochs for environments, which epochs are not time limited
    :param epoch_checkpoint: Checkpoint for printing the learning progress and rendering the environment
    :return: List of cumulative rewards for the episodes
    """

    # Measure the time we need to learn
    time_start = time.time()

    critic_loss = []

    scores_deque = deque(maxlen=epoch_checkpoint)
    e_cumulative_rewards = []
    for e in range(1, epochs + 1):
        state = env.reset()
        agent.reset()
        cumulative_reward = 0
        t = 0
        for t_i in range(max_steps):
            t += 1
            if e % epoch_checkpoint == 0:
                env.render()
            action = agent.act(state)
            # print(action)
            next_state, reward, done, _ = env.step(action)
            # print(reward)
            # reward = reward*1000  # Qube-v0 rewards are VERY small; only for debugging
            if (t_i % update_frequency) == 0:
                perform_update = True
            else:
                perform_update = False
            critic_l = agent.step(state, action, reward, next_state, done, perform_update)
            if critic_l is not None:
                critic_loss.append(critic_l.item())
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
            print('\rEpisode {}\tAverage Reward: {:.2f}\t({:.2f} min elapsed)'.
                  format(e, np.mean(scores_deque), (time.time() - time_start)/60))

    print("Learning weights took {:.2f} min.".format((time.time() - time_start) / 60 ))
    print("Final average cumulative reward", np.mean(e_cumulative_rewards))

    return e_cumulative_rewards


# Plot the cumulative reward per episode
scores = training()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
