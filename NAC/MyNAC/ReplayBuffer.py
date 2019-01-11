
from collections import deque


class ReplayBuffer():
    def __init__(self):
        self.memory = deque(maxlen=2000)


    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])
