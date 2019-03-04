import gym
import numpy as np
import tensorflow as tf

from REINFORCE_ALEX.ReinforceAgent import REINFORCEAgent


def training(render=False):
    env = gym.make('Levitation-v1')
    # ------------------------------------------------ #
    print(env.spec.id)
    print("State Space:\t Shape: {}\t Low: {}\t High {}"
          .format(env.observation_space.shape, env.observation_space.low,
                  env.observation_space.high))
    print("Action Space:\t Shape: {}\t Low: {}\t High {}"
          .format(env.action_space.shape, env.action_space.low,
                  env.action_space.high))
    discrete_actions = np.linspace(start=env.action_space.low,
                                   stop=env.action_space.high,
                                   num=5)
    print("Discrete Actions: ", discrete_actions)
    # ------------------------------------------------ #
    agent = REINFORCEAgent(env, discounting=0.99, learning_rate=0.01, load_weights=False)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    episodes = 10
    for e in range(episodes):
        reward = agent.run_episode(sess, e)


if __name__ == "__main__":
    training()
