import random
import numpy as np
import gym


class ActionDisc(gym.Space):
    """Self defined action space"""
    def __init__(self, high, low, number):
        gym.Space.__init__(self, (), np.float)
        self.high = high
        self.low = low
        self.n = number
        self.space = np.linspace(self.low, self.high, self.n)
        print("Discretizised action space: {}".format(self.space))

    def sample(self):
        return random.choice(self.space)

    def contains(self, x):
        """
        :param x: action space values (as array or similar)
        :return: index of the given actions (as list)
        """
        indices = []
        for i in x:
            indices.append(np.where(self.space==i)[0][0])
        return indices

    def length(self):
        return len(self.space)
