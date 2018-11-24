import tensorflow as tf
import numpy as np
import gym
import quanser_robots
from keras.layers import Dense, Input, Activation
from keras.optimizers import Adam
from keras.models import Model

import random
from collections import deque


class Replay(object):
    def __init__(self, buffer_number):
        self.size = buffer_number
        self.ReplayBuffer = deque()

    def add_observation(self, state, action, reward, next_state, time):
        self.ReplayBuffer.append([state, action, reward, next_state, time])
        if len(self.ReplayBuffer) > self.size:
            self.ReplayBuffer.popleft()

    def random_batch(self, batch_size):
        element_number = len(self.ReplayBuffer)
        if element_number > batch_size:
            batch_size = element_number
        expectations = random.sample(self.ReplayBuffer, k=element_number)
        states = list(zip(*expectations))[0]
        actions = list(zip(*expectations))[1]
        rewards = list(zip(*expectations))[2]
        next_states = list(zip(*expectations))[3]
        times = list(zip(*expectations))[4]
        return states, actions, rewards, next_states, times


class Actor(object):
    def __init__(self, state_size, action_size, learningrate, batch_size, hidden_neurons):
        self.learning = learningrate
        self.state_space = state_size
        self.action_space = action_size
        self.replay_buffer = Replay(5000)
        self.tau = 0.1


    def create_network(self):
        input = Input(self.state_space)
        h1 = Dense(100, activation='relu')(input)
        output = Dense(self.action_space, activation='tanh')(h1)
        nn = Model(input= input, output=output)
        adam_optimizer = Adam(self.learning)
        nn.compile(loss='mse', optimizer=adam_optimizer)