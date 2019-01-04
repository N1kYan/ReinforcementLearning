import gym
import quanser_robots
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from ddpg_agent import Agent

env = gym.make('Pendulum-v0')
env_observation_size = len(env.reset())
env_action_size = len(env.action_space.sample())
env_action_low = env.action_space.low
env_action_high = env.action_space.high
random_seed = 123
env.seed(random_seed)

agent = Agent(state_size=env_observation_size, action_size=env_action_size,
              action_bounds=(env_action_low, env_action_high), random_seed=random_seed)


def training(n_episodes=1000, max_t=300, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    for e in range(1, n_episodes+1):
        state = env.reset()
        agent.reset()
        score = 0
        for t in range(max_t):
            if e % print_every == 0:
                env.render()
            action = agent.act(state)
            # print(action)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        env.close()
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score:{:.2f}'.
              format(e, np.mean(scores_deque)), end="")
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if e % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_deque)))

    return scores


scores = training()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()