import matplotlib.pyplot as plt
import numpy as np

"""
    Evaluation stuff to see the predictions, discretizations and learned functions in action
"""

def evaluate(env, episodes, map_to_state, policy, render):

    rewards_per_episode = []

    print("Evaluating...")

    for e in range(episodes):

        # Discretize first state
        state = env.reset()
        index = map_to_state(state)

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
            index = map_to_state(state)

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
