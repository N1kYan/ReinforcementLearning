import gym
import quanser_robots
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from torch_ddpg.ddpg_agent import Agent

env = gym.make('Pendulum-v0')
env_observation_size = len(env.reset())
env_action_size = len(env.action_space.sample())
env_action_low = env.action_space.low
env_action_high = env.action_space.high
random_seed = 123
env.seed(random_seed)

agent = Agent(state_size=env_observation_size, action_size=env_action_size,
              action_bounds=(env_action_low, env_action_high), random_seed=random_seed)


def training(epochs=1000, max_steps=300, epoch_checkpoint=100):
    scores_deque = deque(maxlen=epoch_checkpoint)
    e_cumulative_rewards = []
    for e in range(1, epochs + 1):
        state = env.reset()
        agent.reset()
        cumulative_reward = 0
        for t in range(max_steps):
            if e % epoch_checkpoint == 0:
                env.render()
            action = agent.act(state)
            # print(action)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            cumulative_reward += reward
            if done:
                break
        env.close()
        scores_deque.append(cumulative_reward)
        e_cumulative_rewards.append(cumulative_reward)
        print('\rEpisode {}\tAverage Reward:{:.2f}'.
              format(e, np.mean(scores_deque)), end="")
        if e % epoch_checkpoint == 0:
            print('\rEpisode {}\tAverage Reward: {:.2f}'.format(e, np.mean(scores_deque)))

    return e_cumulative_rewards


scores = training()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()