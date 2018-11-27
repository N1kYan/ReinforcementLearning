# TODO: Doc
# This discretization can currently be used for the
# ai gym pendulum environment

import numpy as np
from decimal import *

class Discretization:
    
    def __init__(self, name, platform, state_space_size = (9, 17), action_space_size = 9):
        
        # Setting precision for decimals
        getcontext().prec = 6 
        
        self.name = name
        self.platform = platform
        # Determine action space 
        # and sice of state space used for look-up-tables for value function and policy
        if(self.platform=="Pendulum" and self.name=="easy"):
            # Amount of actions in discrete action space, default is 9
            self.action_space_size = action_space_size
            # Discrete action space
            self.action_space = np.linspace(-2,2,self.action_space_size)
            # Amount of states in 3 dim discrete state space, default is (21, 21, 17)
            self.state_space_size = state_space_size
            # 3 dim discrete state space
            self.state_space = (np.linspace(-np.pi, np.pi, self.state_space_size[0]),
                                np.linspace(-8, 8, self.state_space_size[1]))
        else:
            print("Wrong name or platform for discretization object!")
            

    # Maps discrete state to index for value function or policy lookup table
    def map_to_index(self, x):
        if(self.platform=="Pendulum" and self.name=="easy"):
            i1 = int((x[0]+np.pi)/(2*np.pi/(self.state_space_size[0]-1)))
            i2 = int((x[1]+8)/(2*8/(self.state_space_size[1]-1)))
            return np.array([i1, i2])




"""
          This method transforms cos and sin input
          to true degree value by arctan2 function
          
          unused!

"""
def my_arctan(cos, sin):
    return np.rad2deg(np.arctan2(sin, cos))