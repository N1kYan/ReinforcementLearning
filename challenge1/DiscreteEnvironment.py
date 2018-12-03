import numpy as np
import matplotlib.pyplot as plt


class DiscreteEnvironment:
    def __init__(self, env, name, state_space_size, action_space_size):
        self.env = env
        self.state_space_size = state_space_size
        if name == 'EasyPendulum':
            # Pendulum-v2 equidistand
            self.state_space = (np.linspace(-np.pi, np.pi, self.state_space_size[0]),
                            np.linspace(-8, 8, self.state_space_size[1]))
        else:
            print("Unknown discrete environment name.")
        self.action_space_size = action_space_size
        # Currently only 1D actions
        self.action_space = (np.linspace(self.env.action_space.low, self.env.action_space.high,
                                        self.action_space_size[0]),)
        # Transition probabilities
        self.p = np.zeros(shape=(self.state_space_size + self.action_space_size + self.state_space_size))
        self.p_counter = np.zeros(shape=(self.state_space_size + self.action_space_size + self.state_space_size))
        self.action_counter = np.zeros(shape=(self.state_space_size + self.action_space_size))

    # Maps discrete state to index for value function or policy lookup table
    def map_to_state(self, x):
        if len(x) != len(self.state_space_size):
            print("Input shape not matching state space shape.")
            return -1
        # Map every dimension of input to closest value of state space in same dimension
        indices = []
        for i in range(len(x)):
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

    """
        This function updates the environments transition probability function by sampling for 
        'epochs' and counting the results        
    """
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

    """
        This function returns a dictionary of successor states and their probabilities.
        The inputs are indices of a discrete state and a discrete action.
        The output is a python dictionary with state tuples and their probability.
        Returns -1 if no successors sampled (yet)
    """
    def return_successors(self, x, a):
        successors = self.p[x[0], x[1], a, :, :]
        successors_dict = {}
        for s1 in range(successors.shape[0]):
            for s2 in range(successors.shape[1]):
                prob = self.p[x[0], x[1], a, s1, s2]
                if prob != 0:
                    successors_dict.update({(s1, s2): prob})
        if len(successors_dict) == 0:
            return -1
        else:
            return successors_dict


