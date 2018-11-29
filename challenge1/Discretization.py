# TODO: Doc
# This discretization can currently be used for the
# ai gym pendulum environment

from __future__ import print_function
import numpy as np
from decimal import *


# Setting precision for decimals
getcontext().prec = 6

class Discretization:
    def __init__(self, state_space_size, action_space_size):
        # Determine action space
        # and size of state space used for look-up-tables for value function and policy
        # Amount of actions in discrete action space, default is 9
        self.action_space_size = action_space_size
        self.action_space = np.zeros(shape=(self.action_space_size, 1))
        self.state_space_size = state_space_size
        self.state_space = np.zeros(shape=(self.state_space_size))

    # Maps discrete state to index for value function or policy lookup table
    def map_to_index(self, x):
        s0 = self.state_space[0]
        # Get list of distances from x[0] to all elements in state_space[0]
        i0 = min(self.state_space[0], key=lambda c: abs(c-x[0]))
        # Get index of element with min distance
        i0 = np.where(s0 == i0)[0].reshape(-1)[0]

        s1 = self.state_space[1]
        i1 = min(self.state_space[1], key=lambda c: abs(c-x[1]))
        i1 = np.where(s1 == i1)[0].reshape(-1)[0]
        return np.array([i0, i1])



# Discretization with equidistand linspaces
class PendulumDiscretization(Discretization):
    
    def __init__(self, state_space_size, action_space_size):
        super().__init__(state_space_size, action_space_size)
        # Discrete equidistant action space
        self.action_space = np.linspace(-2,2,self.action_space_size)
        # Discrete equidistant 2-dimensional state space
        self.state_space = (np.linspace(-np.pi, np.pi, self.state_space_size[0]),
                            np.linspace(-8, 8, self.state_space_size[1]))



# Handmade Discretization
class EasyPendulumDiscretization(Discretization):

    def __init__(self, state_space_size, action_space_size):
        super().__init__(state_space_size, action_space_size)
        self.action_space = np.linspace(-2, 2, self.action_space_size)

        self.state_space = (np.array([-np.pi, -np.pi * 0.75, -np.pi*0.5, -np.pi*0.25, -0.5, -0.25, -0.1, 0,
                                      0.1, 0.25, 0.5, np.pi*0.25, np.pi*0.5, np.pi*0.75, np.pi]),
                            np.linspace(-8, 8, self.state_space_size[1]))

        self.state_space_size = (np.size(self.state_space[0]), np.shape(self.state_space[1])[0])
