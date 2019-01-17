import gym
import quanser_robots
import numpy as np

import tensorflow as tf
from keras import backend as K


import sys
from MyNAC.ReplayBuffer import ReplayBuffer
from MyNAC.Actor import Actor
from MyNAC.Critic import Critic

""" Natural Actor Critic from Yannik Frisch, Tabea Wilke & Max A. Gehrke"""

UPDATES = 10
EPISODES = 10
N = 200
EPSILON = 0.1
ALPHA = 0.05
GAMMA = 0.9

def main():
    """ Here we implement episodic NAC """

    sess = tf.Session()
    K.set_session(sess)

    env = gym.make("Pendulum-v0")

    print("\n######")
    print(env.spec)
    print("Observation Space: {}".format(env.observation_space.shape))
    print("Action Space: {}".format(env.action_space.shape))

    memory = ReplayBuffer()
    actor = Actor(env=env, sess=sess)
    critic = Critic(env=env, sess=sess)

    state = env.reset()
    print(state.shape)

    w_old = critic.get_weights()
    w_new = ...

    for u in range(1, UPDATES + 1):
        for e in range(1, EPISODES + 1):
            for t in range(1, N + 1):
                action = actor.predict(state)
                next_state, reward, done, info = env.step(action)

                memory.remember(state, action, reward, next_state, done)
                if done:
                    break  # break inner most loop

            # TODO: DO we call the updates after each episode or do we fill \
            # TODO: the buffer with multiple episodes first.

            # CRITIC EVALUATION
            J = None

            # CRITIC UPDATE
            R_e = 0
            Phi_e = ...
            t = 0
            while memory.length() > 0:
                s, a, r, next_s, done = memory.popleft()

                if t == 0:
                    J = critic.predict(s, a)

                R_e += np.math.pow(GAMMA, t) * r
                Phi_e += actor.get_gradients(s)
                t += 1

                w_new_and_J = tf.linalg.inv(Phi_e * Phi_e) * Phi_e * R_e
                w_new, J_test = w_new_and_J[:-1], w_new_and_J[-1]
                assert(J, J_test)
                critic.set_weights(w_new)


            # ACTOR UPDATE
            if angle_between(w_new, w_old) <= EPSILON:
                actor.update(w_new, ALPHA)

            env.reset()

            # TODO: add w_tau later via deque. atm we have a delay of 1
            w_old = w_new


def angle_between(v1, v2):
    """
    Returns the angle in radians between vector 'v1' and 'v2'.

    :param v1: vector (same dimension as v2)
    :param v2: vector (same dimension as v1)
    :return: the angle in radians between vector 'v1' and 'v2'
    """

    # Create the unit vector of the vectors
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)

    # Use cosine trigonometry to calculate the angle
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


if __name__ == "__main__":
    main()
