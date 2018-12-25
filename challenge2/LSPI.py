import gym
import quanser_robots
import numpy as np

env = gym.make("CartpoleStabShort-v0")

episodes = 100

for e in range(episodes):

    state = env.reset()

    timestep = 0

    while True:
        env.render()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        timestep += 1

        if done:
            print("Episode {} finished after {} timesteps".format(e, timestep))
            break
