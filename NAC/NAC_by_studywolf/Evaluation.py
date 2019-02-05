import gym
import time
import numpy as np
import random
import tensorflow as tf
import sys


def evaluate(env, policy_grad, episodes, render, sleep, sess):
    time_steps = 10000

    print("\nEVALUATION: {} episodes with {} time steps each (or until 'done')"
          .format(episodes, time_steps))

    # unpack the policy network (generates control policy)
    (pl_calculated, pl_state, pl_actions,
        pl_advantages, pl_optimizer) = policy_grad

    for e in range(episodes):
        print("Episode {} ... ".format(e), end='')
        sys.stdout.flush()
        done = False
        observation = env.reset()
        for t in range(time_steps):
            # Render environment
            if render:
                env.render()
                time.sleep(sleep)

            if done:
                print("held stick for {} time steps!".format(t))
                break

            obs_vector = np.expand_dims(observation, axis=0)
            probs = sess.run(
                pl_calculated,
                feed_dict={pl_state: obs_vector})

            # Check which action to take
            # stochastically generate action using the policy output
            probs_sum = 0
            action_i = None
            rnd = random.uniform(0, 1)
            for k in range(len(env.action_space)):
                probs_sum += probs[0][k]
                if rnd < probs_sum:
                    action_i = k
                    break

            action = env.action_space[action_i]
            if env.name == 'DoublePendulum-v0':
                action = np.array([action])

            observation, reward, done, info = env.step(action)



