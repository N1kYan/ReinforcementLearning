import random
import numpy as np
from collections import deque
import torch
from torch.autograd import Variable


class MemoryDQN(object):
    def __init__(self, buffer_number):
        self.size = buffer_number
        self.ReplayBuffer = deque()

    def add_observation(self, state, action, reward, next_state):
        self.ReplayBuffer.append([state, action, reward, next_state])
        if len(self.ReplayBuffer) > self.size:
            self.ReplayBuffer.popleft()

    def random_batch(self, batch_size):
        element_number = len(self.ReplayBuffer)
        if element_number < batch_size:
            batch_size = element_number
        expectations = random.sample(self.ReplayBuffer, k=batch_size)
        states = list(zip(*expectations))[0]
        actions = list(zip(*expectations))[1]
        rewards = list(zip(*expectations))[2]
        next_states = list(zip(*expectations))[3]
        return np.array(states), np.array(actions), \
               torch.from_numpy(np.array(rewards)).type(torch.FloatTensor),np.array(next_states)

    def size_mem(self):
        return len(self.ReplayBuffer)
