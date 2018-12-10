from __future__ import print_function
import numpy as np
from scipy.stats import multivariate_normal
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
            # Pendulum-v2 equidistant
            self.amp = [np.pi, 8]
            self.state_space = (np.linspace(-self.amp[0], self.amp[0], self.state_space_shape[0]),
                                np.linspace(-self.amp[1], self.amp[1], self.state_space_shape[1]))
            self.number_states = np.prod(self.state_space_shape)

        elif name == 'LowerBorder' or name == 'UpperBorder':
            self.state_space = (np.array([-np.pi, -np.pi*(2/3), -np.pi*(1/3), -0.8, -0.6, -0.4, -0.2, -0.15, -0.1,
                                          -0.05, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, np.pi*(1/3), np.pi*(2/3), np.pi]),
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

    # Maps discrete state to index for value function or policy lookup table
    def map_to_state(self, x):

        if len(x) != len(self.state_space_shape):
            print("Input shape not matching state space shape.")
            return -1

        indices = []

        # Clamp to values on a circle... (2-pi periodic)
        # only for first dim of pendulum
        if x[1] > self.amp[1]:
            x[1] = x[1] - 2 * self.amp[1]
        elif x[1] < -self.amp[1]:
            x[1] = x[1] + 2 * self.amp[1]

        for dim in range(len(x)):
            if self.name == 'LowerBorder':
                for s in np.arange(0, len(self.state_space[dim]) - 1):
                    if x[dim] <= 0:
                        if self.state_space[dim][s] <= x[dim] < self.state_space[dim][s+1]:
                            indices.append(s)
                    elif x[dim] > 0:
                        if self.state_space[dim][s] < x[dim] <= self.state_space[dim][s+1]:
                            indices.append(s+1)
            elif self.name == 'EasyPendulum':
                # Map every dimension of input to closest value of state space in same dimension
                s = self.state_space[dim]
                # Get list of distances from x[0] to all elements in state_space[0]
                index = min(self.state_space[dim], key=lambda c: abs(c-x[dim]))
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

    # Multivariate gaussian of input 'x'
    # x : Input vector, N dimensional
    # mean: Mean vector for N dimensions
    # sigma : NxN covariance matrix
    def _gaussian(self, x, mean, sigma):
        return multivariate_normal.pdf(x=x, mean=mean, cov=sigma)

    # Return all states inside the [-3sig,+3sig] interval from a (multivariate) gaussian
    # with mean set as the current state
    def get_successors(self, state, action, sigmas):
        # List of successors (dictionary)
        # holding index of discrete state as key
        # and list of [probability, reward] of this state as value
        successor_list = {(0, 0): np.array([0, 0])}
        regression_input = np.concatenate([state, action]).reshape((1, -1))
        next_state = self.regressorState.predict(regression_input)[0]
        mean = np.copy(next_state)

        max_index = next_state + np.dot(3, sigmas)
        min_index = next_state - np.dot(3, sigmas)
        # Euklidean distance
        dist = np.linalg.norm(max_index - min_index)
        # Granularity of 3 sigma intervall
        granularity = 20
        # TODO: Modular for n dim states
        print("State: ", next_state)
        print("Discrete state: ", self.map_to_state(next_state))
        print("3 sigma: ", np.dot(3, sigmas))
        print("State - 3 sigma: {}, State + 3 sigma: {}".format(min_index, max_index))
        print()
        # So far so good




        for a in np.arange(min_index[0], max_index[0], step=1.0/granularity):
            for b in np.arange(min_index[1], max_index[1], step=1.0/granularity):
                # List holding state, index, prob, reward
                successor_state = [a, b]
                successor_index = self.map_to_state(successor_state)
                successor_probability = self._gaussian(successor_state, mean=mean, sigma=sigmas)
                successor_reward = self.regressorReward.predict(regression_input)[0]

                # If state index already in list, add probabilites
                if (successor_index[0], successor_index[1]) not in successor_list:
                    successor_list.update({(successor_index[0], successor_index[1]):
                                               np.array([successor_probability, successor_reward])})
                else:
                    # Get index of successor state id in list of successors
                    # Approximating Riemann sum
                    successor_list[(successor_index[0], successor_index[1])][0] += successor_probability\
                                                                                   *(dist/granularity)
        print("List of all successors: \n", successor_list)
        return np.array(successor_list)

    # Creates regressor object and performs regression
    # Returns a regressor for the state and one for the reward
    def perform_regression(self, env, epochs, save_flag):
        reg = Regressor()
        self.regressorState, self.regressorReward = reg.perform_regression(env=env, epochs=epochs, save_flag=save_flag)
