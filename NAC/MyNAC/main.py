import gym
import quanser_robots

import sys
from MyNAC.ReplayBuffer import ReplayBuffer
from MyNAC.Actor import Actor

UPDATES = 10
EPISODES = 10
N = 200

def main():
    env = gym.make("Pendulum-v0")

    print("\n######")
    print(env.spec)
    print("Observation Space: {}".format(env.observation_space.shape))
    print("Action Space: {}".format(env.action_space.shape))

    memory = ReplayBuffer()
    actor = Actor(env)

    state = env.reset()
    print(state.shape)

    for u in range(1, UPDATES + 1):
        for e in range(1, EPISODES + 1):
            for t in range(1, N + 1):
                action = actor.predict(state)
                next_state, reward, done, info = env.step(action)

                memory.remember(state, action, reward, next_state, done)
                if done:
                    break  # break only one loop

            env.reset()




if __name__ == "__main__":
    main()
