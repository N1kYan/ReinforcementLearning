import gym
import time
import numpy as np
import random
import tensorflow as tf


def evaluate(env, policy_grad, episodes, render, sleep, sess):
    time_steps = 100000

    print("\nEVALUATION: {} episodes with {} time steps each (or until 'done')"
          .format(episodes, time_steps))

    # unpack the policy network (generates control policy)
    (pl_calculated, pl_state, pl_actions,
        pl_advantages, pl_optimizer) = policy_grad

    for e in range(episodes):
        done = False
        observation = env.reset()
        for t in range(time_steps):
            # Render environment
            if render:
                env.render()
                time.sleep(sleep)

            if done:
                break

            obs_vector = np.expand_dims(observation, axis=0)
            probs = sess.run(
                pl_calculated,
                feed_dict={pl_state: obs_vector})

            # stochastically generate action using the policy output
            action = 0 if random.uniform(0, 1) < probs[0][0] else 1

            observation, reward, done, info = env.step(action)



