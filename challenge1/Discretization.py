# TODO: Doc
# This deiscretization can currentyl be used for the 
# ai gym pendulum environment

import numpy as np
from decimal import *

class Discretization:
    
    
    
    def __init__(self, name, platform):
        
        # Setting precision for decimals
        getcontext().prec = 6 
        
        self.name = name
        self.platform = platform
        # Determine action space 
        # and sice of state space used for look-up-tables for value function and policy
        if(self.platform=="Pendulum"):
            self.action_space_size = 8+1
            self.action_space = np.linspace(-2,2,self.action_space_size)
            self.state_space_size = (21, 21, 17)
            self.state_space = (np.linspace(-1, 1, self.state_space_size[0]),
                                np.linspace(-1, 1, self.state_space_size[1]),
                                np.linspace(-8, 8, self.state_space_size[2]))

    # Maps discrete state to index for value function or policy lookup table
    def map_to_index(self, x):
        if(self.platform=="Pendulum"):
            index = [int(10*(Decimal(x[0])+1)), int(10*(Decimal(x[1])+1)), int(Decimal(x[2])+8)]
            return index
    
    # Maps index back to state
    def map_to_state(self, x):
        if(self.platform=="Pendulum"):
            state = [(x[0]/10)-1, (x[1]/10)-1, x[2]-8]
            return state
       