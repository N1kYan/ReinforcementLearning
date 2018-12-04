import numpy as np
import matplotlib.pyplot as plt
from Regression import Regressor

class DiscreteEnvironment:
    def __init__(self, env, name, state_space_shape, action_space_shape):
        self.env = env
        self.name = name

        if name == 'EasyPendulum':
            self.state_space_shape = state_space_shape
            # Pendulum-v2 equidistand
            self.state_space = (np.linspace(-np.pi, np.pi, self.state_space_size[0]),
                            np.linspace(-8, 8, self.state_space_size[1]))
            self.number_states = np.prod(self.state_space_size)

        elif name == 'LowerBorder' or name == 'UpperBorder':
            self.state_space = (np.array([-np.pi, -np.pi*(2/3), -np.pi*(1/3), -0.8, -0.6, -0.4, -0.2, -0.15, -0.1, -0.05,
                                 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, np.pi*(1/3), np.pi*(2/3), np.pi]),
                                np.array([-8.0, -7.0, -6.0, -5.0, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.25,
                                 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0]))
            self.state_space_size = (len(self.state_space[0]), len(self.state_space[1]))
            self.number_states = np.prod(self.state_space_size)

        else:
            print("Unknown discrete environment name.")

        self.action_space_size = action_space_shape
        # Currently only 1D actions
        self.action_space = (np.linspace(self.env.action_space.low, self.env.action_space.high,
                                        self.action_space_size[0]),)
        self.number_actions = np.prod(self.action_space_size)

        # Initialize transition probabilities
        # Will later hold entries with successor states, their probabilities and their reward
        # [[prob_next_s, next_s, reward], ...]
        self.P = np.zeros([self.number_states, self.number_actions])

    # TODO: change to 1d arrays for discrete state space
    # Maps discrete state to index for value function or policy lookup table
    def map_to_state(self, x):

        if len(x) != len(self.state_space_size):
            print("Input shape not matching state space shape.")
            return -1

        indices = []

        for i in range(len(x)):
            for s in np.arange(0, len(self.state_space[i])-1):
                if self.name=='LowerBorder':
                    if x[i] <= 0:
                        #print("{} <= {} < {} : {}".format(self.state_space[i][s],
                        #                                   x[i], self.state_space[i][s+1],
                        #                                   self.state_space[i][s] <= x[i] < self.state_space[i][s+1]))
                        if self.state_space[i][s] <= x[i] < self.state_space[i][s+1]:
                            indices.append(s)
                    elif x[i] > 0:
                        if self.state_space[i][s] < x[i] <= self.state_space[i][s+1]:
                            indices.append(s+1)
                elif self.name=='EasyPendulum':
                    # Map every dimension of input to closest value of state space in same dimension
                    s = self.state_space[i]
                    # Get list of distances from x[0] to all elements in state_space[0]
                    index = min(self.state_space[i], key=lambda c: abs(c-x[i]))
                    # Get index of element with min distance
                    index = np.where(s == index)[0].reshape(-1)[0]
                    indices.append(index)
        return np.array(indices)

    # Maps action
    def map_to_action(self, x):
        if len(x) != len(self.action_space_size):
            print("Input shape not matching action space shape.")
            return -1
        indices = []
        for i in range(len(x)):
            s = self.action_space[i]
            # Get list of distances from x[0] to all elements in state_space[0]
            index = min(self.action_space[i], key=lambda c: abs(c-x[i]))
            # Get index of element with min distance
            index = np.where(s == index)[0].reshape(-1)[0]
            indices.append(index)
        return np.array(indices)

    def evaluate_transition_prob(self, env, epochs, save_flag):
        reg = Regressor()
        regressorState, regressorReward = reg.perform_regression(env=env, epochs=epochs, save_flag=save_flag)


    """
        # This function updates the environments transition probability function by sampling for 
        # 'epochs' and counting the results        
    
    def update_transition_probabilities(self, policy, epochs):

        state = self.env.reset()

        for i in range(epochs):
            discrete_state = self.map_to_state(state)
            # TODO: make modular for different state and action spaces
            if policy is None:
                action = self.env.action_space.sample()
            else:
                action = [policy[discrete_state[0], discrete_state[1]]]
            discrete_action = self.map_to_action(action)
            new_state, reward, done, _ = self.env.step(action)
            new_discrete_state = self.map_to_state(new_state)

            index = np.concatenate((discrete_state, discrete_action, new_discrete_state))
            # TODO: make modular for different state and action spaces
            self.action_counter[index[0], index[1], index[2]] += 1
            self.p_counter[index[0], index[1], index[2], index[3], index[4]] += 1

            prob = self.p_counter[index[0], index[1], index[2], index[3], index[4]] / \
                self.action_counter[index[0], index[1], index[2]]
            self.p[index[0], index[1], index[2], index[3], index[4]] = prob

            state = new_state
            if done:
                break

        # TODO: Make modular for all inputs
        for s1 in range(self.p.shape[0]):
            for s2 in range(self.p.shape[1]):
                for a in range(self.p.shape[2]):
                    for ns1 in range(self.p.shape[3]):
                        for ns2 in range(self.p.shape[4]):
                            if self.action_counter[s1, s2, a] == 0:
                                prob = 0
                            else:
                                prob = self.p_counter[s1, s2, a, ns1, ns2] / self.action_counter[s1, s2, a]
                            self.p[s1, s2, a, ns1, ns2] = prob

    
        # This function returns a dictionary of successor states and their probabilities.
        # The inputs are indices of a discrete state and a discrete action.
        # The output is a python dictionary with state tuples and their probability.
        # Returns -1 if no successors sampled (yet)
    
    def return_successors(self, x, a):
        successors = self.p[x[0], x[1], a, :, :]
        successors_dict = {}
        for s1 in range(successors.shape[0]-1):
            for s2 in range(successors.shape[1]-1):
                print(x[0], x[1], a, s1, s2)
                prob = self.p[x[0], x[1], a, s1, s2]
                if prob != 0:
                    successors_dict.update({(s1, s2): prob})
        if len(successors_dict) == 0:
            return -1
        else:
            return successors_dict
    """


