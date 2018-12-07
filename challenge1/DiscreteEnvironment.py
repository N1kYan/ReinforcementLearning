from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from Regression import Regressor

class DiscreteEnvironment:
    def __init__(self, env, name, state_space_shape, action_space_shape):
        self.env = env
        self.name = name
        self.regressorState = None
        self.regressorReward = None

        if name == 'EasyPendulum':
            self.state_space_shape = state_space_shape
            # Pendulum-v2 equidistand
            self.amp = [np.pi, 8]
            self.state_space = (np.linspace(-self.amp[0], self.amp[0], self.state_space_shape[0]),
                            np.linspace(-self.amp[1], self.amp[1], self.state_space_shape[1]))
            self.number_states = np.prod(self.state_space_shape)

        elif name == 'LowerBorder' or name == 'UpperBorder':
            self.state_space = (np.array([-np.pi, -np.pi*(2/3), -np.pi*(1/3), -0.8, -0.6, -0.4, -0.2, -0.15, -0.1, -0.05,
                                 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, np.pi*(1/3), np.pi*(2/3), np.pi]),
                                np.array([-8.0, -7.0, -6.0, -5.0, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.25,
                                 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0]))
            self.state_space_shape = (len(self.state_space[0]), len(self.state_space[1]))
            self.number_states = np.prod(self.state_space_shape)

        else:
            print("Unknown discrete environment name.")

        self.action_space_shape = action_space_shape
        # Currently only 1D actions
        self.action_space = (np.linspace(self.env.action_space.low, self.env.action_space.high,
                                        self.action_space_shape[0]),)
        self.number_actions = np.prod(self.action_space_shape)

        # Initialize transition probabilities
        # Will later hold entries with successor states, their probabilities and their reward
        # [[prob_next_s, next_s, reward], ...]
        self.P = np.zeros([self.state_space_shape[0], self.state_space_shape[1], self.action_space_shape[0]])

    # TODO: change to 1d arrays for discrete state space
    # Maps discrete state to index for value function or policy lookup table
    def map_to_state(self, x):

        if len(x) != len(self.state_space_shape):
            print("Input shape not matching state space shape.")
            return -1

        indices = []

        for i in range(len(x)):

            # Clamp to values on a circle... (2-pi periodic)
            # only for first dim of pendulum
            if x[1] > self.amp[1]:
                x[1] = x[1]-2*self.amp[1]
            elif x[1] < -self.amp[1]:
                x[1] = x[1]+2*self.amp[1]

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
        if len(x) != len(self.action_space_shape):
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

    def gaussian_function(self, x, mean, sigma):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * np.square((x - mean) / sigma))

    def gaussian(self, x, mean, sigmas):
        returns = []
        for c in range(len(x)):
            returns.append(self.gaussian_function(x[c], mean[c], sigmas[c]))
        return returns

    # Return all states inside the [-3sig,+3sig] interval from a (multivariate) gaussian
    # with mean set as the current state
    def get_successors(self, state, action, sigmas):
        granularity = 10
        successors = []
        regression_input = np.concatenate([state, action]).reshape((1, -1))
        next_state = self.regressorState.predict(regression_input)[0]
        # (n dimensional) Index of +3sigma
        mean = np.copy(next_state)
        max_index = next_state + np.dot(3, sigmas)
        min_index = next_state - np.dot(3, sigmas)
        # TODO: Modular for n dim states
        print("state: ", next_state)
        print("3 sigma: ", np.dot(3, sigmas))
        print("State - 3 sigma: {}, State + 3 sigma: {}".format(min_index, max_index))
        for a in np.arange(min_index[0], max_index[0], granularity):
            for b in np.arange(min_index[1], max_index[0], granularity):
                state = np.array([a, b])
                print(state)
                successor = np.array([self.gaussian(state, mean, sigmas), self.map_to_state(state),
                                      self.regressorReward.predict(state.reshape((1, -1)))])
                print(successor)
                if successor not in successors:
                    successors.append(successor)
        print(successors)

        """
        for s1 in np.arange(min_index[0], max_index[0], self.granularity[0]):
            for s2 in np.arange(min_index[1], max_index[1], self.granularity[1]):
                x = np.array([s1, s2])
                index = self.map_to_state(x)
                prob = self.gaussian(x, mean, sigmas)
                reward = self.regressorReward.predict(regression_input)
                successors.append([index, prob, reward])
        """
        return np.array(successors)

    # Creates regressor object and performs regression
    # Returns a regressor for the state and one for the reward
    def perform_regression(self, env, epochs, save_flag):
        reg = Regressor()
        self.regressorState, self.regressorReward = reg.perform_regression(env=env, epochs=epochs, save_flag=save_flag)
