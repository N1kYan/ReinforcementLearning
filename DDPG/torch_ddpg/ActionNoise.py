import numpy as np
import random
import copy


# TODO: Source, descriptions
class OUNoise:
    """
        Ornstein Uhlenbeck noise for actions.

    """
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=1.2):
        """
        Initializes the parameters and the noise process.
        :param size: Dimensions of the environments actions
        :param seed: Random seed
        :param mu: Mean for OU process (default = 0)
        :param theta: TODO
        :param sigma: Variance for OU process
            0.2 works well for Pendulum-v0; 1.2 for Qube-v0?
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """
        Resets the noise process' internal state to mean mu.
        :return: None
        """
        self.state = copy.copy(self.mu)

    def sample(self):
        """
        Updates the noise process' internal state and returns it as noise sample.
        :return: The noise sample
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
