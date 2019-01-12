
from collections import deque


class ReplayBuffer():
    def __init__(self):
        self.memory = deque()

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])

    def reset(self):
        self.memory.clear()

    def popleft(self):
        return self.memory.popleft()

    def length(self):
        return len(self.memory)
