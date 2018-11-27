# TODO: Doc
# This deiscretization can currentyl be used for the 
# ai gym pendulum environment

import numpy as np
from decimal import *

class Discretization:
    
    def __init__(self, name, platform, state_space_size = (21, 21, 17), action_space_size = 9):
        
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
            self.state_space = (np.linspace(-1, 1, self.state_space_size[0]),
                                np.linspace(-1, 1, self.state_space_size[1]),
                                np.linspace(-8, 8, self.state_space_size[2]))

        if(self.platform=="Pendulum" and self.name=="degree_only"):
            # Amount of actions in discrete action space, default is 9
            self.action_space_size = action_space_size
            # Discrete action space
            self.action_space = np.linspace(-2,2,self.action_space_size)
            # Amount of states in 3 dim discrete state space, default is (21, 21, 17)
            self.state_space_size = state_space_size
            # 3 dim discrete state space
            self.state_space = (np.linspace(-180, 180, self.state_space_size[0]),
                                np.linspace(-8, 8, self.state_space_size[1]))
                
        else:
            print("Wrong name or platform for discretization object!")
            

    # Maps discrete state to index for value function or policy lookup table
    def map_to_index(self, x):
        if (self.platform=="Pendulum" and self.name=="degree_only"):
            # Map to positive state space
            i1 = Decimal(x[0])+180
            # Divide by 'frequency' of first state space dimension
            i1 = i1/(360/(Decimal(self.state_space_size[0])-1))
            # Round to integer
            i1 = int(i1)
            # Analogue with angular velocity
            i2 = Decimal(x[1]) + 8
            i2 = i2/(16/(Decimal(self.state_space_size[1])-1))
            i2 = int(i2)
            index = [i1, i2]
            return index
        
        if(self.platform=="Pendulum" and self.name=="easy"):
            index = [int(10*(Decimal(x[0])+1)), int(10*(Decimal(x[1])+1)),
                     int(Decimal(x[2]+(self.state_space_size[2]-1)/2))]
            return index




"""
          This method transforms cos and sin input
          to true degree value by arctan2 function

"""
def my_arctan(cos, sin):
    return np.rad2deg(np.arctan2(sin, cos))