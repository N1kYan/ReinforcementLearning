# TODO: Doc
# This discretization can currently be used for the
# ai gym pendulum environment

from __future__ import print_function
import numpy as np
from decimal import *


# Setting precision for decimals
getcontext().prec = 6


class PendulumDiscretization:
    
    def __init__(self, state_space_size=(9, 17), action_space_size=9):
        # Determine action space 
        # and sice of state space used for look-up-tables for value function and policy
        # Amount of actions in discrete action space, default is 9
        self.action_space_size = action_space_size
        # Discrete action space
        self.action_space = np.linspace(-2,2,self.action_space_size)
        # Amount of states in 3 dim discrete state space, default is (21, 21, 17)
        self.state_space_size = state_space_size
        # 3 dim discrete state space
        self.state_space = (np.linspace(-np.pi, np.pi, self.state_space_size[0]),
                            np.linspace(-8, 8, self.state_space_size[1]))

    # Maps discrete state to index for value function or policy lookup table
    def map_to_index(self, x):
            i1 = int((x[0]+np.pi)/(2*np.pi/(self.state_space_size[0]-1)))
            i2 = int((x[1]+8)/(2*8/(self.state_space_size[1]-1)))
            return np.array([i1, i2])

class EasyPendulumDiscretization:

    def __init__(self, state_space_size=(13, 17), action_space_size=9):
        self.action_space_size = action_space_size
        self.action_space = np.linspace(-2, 2, self.action_space_size)
        self.state_space_size = state_space_size
        self.state_space = (np.array([-np.pi, -np.pi * 0.75, -np.pi*0.5, -np.pi*0.25, -0.5, -0.25, -0.1,
                                      0.1, 0.25, 0.5, np.pi*0.25, np.pi*0.5, np.pi*0.75, np.pi]),
                            np.linspace(-8, 8, self.state_space_size[1]))

    # Maps discrete state to index for value function or policy lookup table
    def map_to_index(self, x):
        if x[0] <= np.pi:
            i1 = 12
            if x[0] <= np.pi*0.75:
                i1 = 11
                if x[0] <= np.pi*0.5:
                    i1 = 10
                    if x[0] <= np.pi * 0.25:
                        i1 = 9
                        if x[0] <= 0.5:
                            i1 = 8
                            if x[0] <= 0.25:
                                i1 = 7
                                if x[0] <= 0.1:
                                    i1 = 6
                                    if x[0] <= -0.1:
                                        i1 = 5
                                        if x[0] <= -0.25:
                                            i1 = 4
                                            if x[0] <= -0.5:
                                                i1 = 3
                                                if x[0] <= -np.pi * 0.25:
                                                    i1 = 2
                                                    if x[0] <= -np.pi * 0.5:
                                                        i1 = 1
                                                        if x[0] <= -np.pi * 0.75:
                                                            i1 = 0
        i2 = int((x[1] + 8) / (2 * 8 / (self.state_space_size[1] - 1)))
        return np.array([i1, i2])




"""
          This method transforms cos and sin input
          to true degree value by arctan2 function
          
          unused!

"""
def my_arctan(cos, sin):
    return np.rad2deg(np.arctan2(sin, cos))