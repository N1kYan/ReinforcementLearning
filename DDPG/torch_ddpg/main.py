import gym
import quanser_robots
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time

from torch_ddpg.DDPGAgent import Agent

env = gym.make('Qube-v0')
# env = gym.make('Pendulum-v0')
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
env.seed(random_seed)

agent = Agent(state_size=env_observation_size, action_size=env_action_size,
              action_bounds=(env_action_low, env_action_high), random_seed=random_seed)


def training(epochs=10000, max_steps=500, epoch_checkpoint=500):
    """
    Runs the training process on the gym environment.
    :param epochs: Number of epochs for training
    :param max_steps: Maximum time-steps for each training epoch;
     Does end epochs for environments, which epochs are not time limited
    :param epoch_checkpoint: Checkpoint for printing the learning progress and rendering the environment
    :return: List of cumulative rewards for the episodes
    """

    # Measure the time we need to learn
    time_learning_start = time.clock()

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
            agent.step(state, action, reward, next_state, done)
            state = next_state
            cumulative_reward += reward
            if done:
                break
        env.close()
        scores_deque.append(cumulative_reward)
        e_cumulative_rewards.append(cumulative_reward)
        print('\rEpisode {}\tAverage Reward: {}\tSteps: {}'.
              format(e, np.mean(scores_deque), t), end="")
        if e % epoch_checkpoint == 0:
            # Print cumulative reward per episode averaged over #epoch_checkpoint episodes
            print('\rEpisode {}\tAverage Reward: {:.2f}\tSteps: {}'.format(e, np.mean(scores_deque), t))

    time_learning_elapsed = (time.clock() - time_learning_start)
    print("Learning the weights took {} secs.".format(time_learning_elapsed))

    return e_cumulative_rewards


# Plot the cumulative reward per episode
scores = training()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
