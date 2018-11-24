import tensorflow as tf
import numpy as np
import gym
import quanser_robots
import torch.nn as nn
import torch.nn.functional as F
import torch
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


class Actor(nn.Module):
    def __init__(self, state_size, action_size, learningrate, batch_size, hidden_neurons, action_limit):
        self.learning = learningrate
        self.state_space = state_size
        self.action_space = action_size
        self.replay_buffer = Replay(5000)
        #self.tau = 0.1

        self.action_limit = action_limit

        self.input = nn.Linear(self.state_space, hidden_neurons)
        self.input.weight.data.uniform_(-0.05, 0.05)

        self.hidden = nn.Linear(hidden_neurons, self.action_space)
        self.hidden.weight.data.uniform_(-0.05,0.05)

    def forward(self, state):
        f1 = F.relu(self.input(state))
        f2 = F.relu(self.hidden(f1))
        out = F.tanh(self.output(f2))

        out = out*self.action_limit
        return out


class Critic(nn.Module):
    def __init__(self, state_size, action_size, learningrate, batch_size, hidden_neurons, action_limit):
        self.learning = learningrate
        self.state_space = state_size
        self.action_space = action_size
        self.replay_buffer = Replay(5000)
        #self.tau = 0.1

        self.action_limit = action_limit

        self.input = nn.Linear(self.state_space, hidden_neurons)
        self.input.weight.data.uniform_(-0.05, 0.05)

        self.hidden = nn.Linear(hidden_neurons+ self.action_space, hidden_neurons)
        self.hidden.weight.data.uniform_(-0.05,0.05)

        self.out = nn.Linear(hidden_neurons, 1)
        self.out.weight.data.uniform_(-0.05,0.05)


    def forward(self, state, action):
        f1 = F.relu(self.input(state))

        x = torch.cat((f1, action),1)
        x = self.hidden(x)
        out = F.relu(x)

        out = self.out(out)
        return out
