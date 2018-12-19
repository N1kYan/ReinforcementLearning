import random
from collections import deque


class ReplayBuffer(object):
    def __init__(self, buffer_number):
        self.size = buffer_number
        self.ReplayBuffer = deque()

    def add_observation(self, state, action, reward, next_state, time):
        self.ReplayBuffer.append([state, action, reward, next_state, time])
        if len(self.ReplayBuffer) > self.size:
            self.ReplayBuffer.popleft()

    def random_batch(self, batch_size):
        element_number = len(self.ReplayBuffer)
        if element_number > batch_size:  # if batch_size > element_number?
            batch_size = element_number
        expectations = random.sample(self.ReplayBuffer, k=element_number)  # k=batch_size?
        states = list(zip(*expectations))[0]
        actions = list(zip(*expectations))[1]
        rewards = list(zip(*expectations))[2]
        next_states = list(zip(*expectations))[3]
        times = list(zip(*expectations))[4]
        return states, actions, rewards, next_states, times
