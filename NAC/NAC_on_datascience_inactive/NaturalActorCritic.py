import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K
from collections import deque

import tensorflow as tf
import random

class NaturalActorCritic:

    def __init__(self, env, sess):
        self.env = env
        self.sess = sess

        # Learning rates
        self.alpha = 0.1
        self.beta = 0.1
        self.gamma = 0.1
        self.delta = 0.1

        # Other parameters
        self.training_epochs = 1000

    def draw_initial_state(self):
        # TODO: define initial state distribution modular for possible environments
        return 0

    # TODO: How to calculate policy network gradient? Should return np.array
    def train(self, env, policy, policy_der, sigma):
        # Initial state and parameters
        state = self.draw_initial_state()
        A = 0
        z = 0
        b = z

        for t in range(self.training_epochs):
            # Draw action from policy network and observe new state and reward
            action = policy(state)
            new_state, reward, done, infos = env.step(action)

            # Critic Evaluation

            # Update basis functions
            sigma_tilde = np.array([sigma(new_state).T, 0]).T
            sigma_hat = np.array([sigma(state).T, policy_der(state).T]).T

            # Statistics
            # TODO: Define delta and gamma; What are their meanings?
            z = self.delta * z + sigma_hat
            A = A + z * (sigma_hat - self.gamma * sigma_tilde).T
            b = b + z * reward

            # Critic parameters
            vec = (np.invert(A) * b).T
            w = vec[0].T
            v = vec[1].T

            # Actor update
            # TODO: Define alpha and beta; What are their meanings?
            theta = theta = self.alpha * w
            z = self.beta * z * A
        return policy