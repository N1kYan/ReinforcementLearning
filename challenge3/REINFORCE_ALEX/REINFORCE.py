import numpy as np
import gym
import quanser_robots
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import sys

class ReinforceAgent:
    def __init__(self, state_space_size, action_space, learning_rate, discounting, load_weights):
        self.state_space_size = state_space_size
        self.action_space = action_space
        self.load_weights_flag = load_weights
        self.gamma = discounting
        self.learning_rate = learning_rate
        self.model = self.init_model()
        self.model.summary()
        self.learning_rate = learning_rate
        self.trajectory_states, self.trajectory_actions, \
            self.trajectory_probs, self.trajectory_rewards \
            = [], [], [], []
        if self.load_weights_flag:
            self.model.load_weights("./saved_models/REINFORCE")

    def init_model(self):
        """
        Creates the keras model of the nn policy approximation.
        :return: Keras model
        """
        model = Sequential()
        model.add(Dense(10, input_dim=self.state_space_size, activation='relu',
                        kernel_initializer='glorot_uniform'))
        model.add(Dense(10, activation='relu',
                        kernel_initializer='glorot_uniform'))
        model.add(Dense(self.action_space.size, activation='softmax',
                        kernel_initializer='glorot_uniform'))

        # Using categorical crossentropy as a loss is a trick to easily
        # implement the policy gradient. Categorical cross entropy is defined
        # H(p, q) = sum(p_i * log(q_i)). For the action taken, a, you set
        # p_a = advantage. q_a is the output of the policy network, which is
        # the probability of taking the action a, i.e. policy(s, a).
        # All other p_i are zero, thus we have H(p, q) = A * log(policy(s, a))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss="categorical_crossentropy", optimizer=opt)
        return model

    def remember(self, state, action, prob, reward):
        """
        Saving a trajectory.
        :param state: current state s of the environment
        :param action: current performed action a
        :param reward: current observed reward r
        :return: none
        """
        self.trajectory_states.append(state)
        self.trajectory_actions.append(action)
        self.trajectory_probs.append(prob)
        self.trajectory_rewards.append(reward)

    def predict_action(self, state):
        """
        Forward pass through the nn policy network.
        Approximating a ~ pi(a|s) and then random sampling from disc. action space.
        :param state: Current state s of the observation.
        :return: Predicted action a, probability of ALL POSSIBLE actions (bitte, danke, Alex)
        """
        state = state.reshape([1, state.shape[0]])
        action_probs = self.model.predict(state, batch_size=1).flatten()
        action = np.random.choice(self.action_space, 1, p=action_probs)[0]
        # return action, action_probs[np.where(self.action_space == action)]

        action_index = np.where(self.action_space==action)[0][0]
        prob_of_action = action_probs[action_index]

        # action = scalar; action_probs = array with probabilities for all
        # possible action
        return action, prob_of_action

    def discounted_rewards(self, rewards):
        """
        Calculate discounted rewards from trajectory.
        :param rewards: Remembered rewards from trajectory (per time-step)
        :return:
        """
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            # The following condition makes no sense to me (Alex)
            # if rewards[t] != 0:
            #     running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self):
        """
        Calculate loss and weight update.
        Update parameters
        :return:
        """

        # REWARDS
        # We have our memory filled with all the rewards from the episode.
        # And our discount function calculates the rewards for every step.
        # So we only need to call it once and not in every iteration.
        # Instead we access the following array with t each iteration
        disc_rewards = self.discounted_rewards(self.trajectory_rewards)

        # Iterate over trajectory
        for t in range(0, len(self.trajectory_states)):
            # ------------------------ π(a|s) -------------------------- #
            # Scalar
            pi_a_s = self.trajectory_probs[t]

            # ----------------------- log(π(a|s)) ----------------------- #
            # Scalar
            log_pi_a_s = K.log(pi_a_s)

            # ------------------- ∇_θ log(π(a|s)) ----------------------- #

            train_weights = tf.reshape(self.model.trainable_weights[0], (-1,))

            print(train_weights)

            log_gradient = tf.gradients(log_pi_a_s, train_weights)

            print("Log gradient:", log_gradient)

            sys.exit()
            # Update parameters
            self.model.set_weights(self.model.trainable_weights + self.learning_rate
                                   * log_gradient * disc_rewards[t])

        # Reset memory
        self.trajectory_states, self.trajectory_actions, self.trajectory_probs, self.trajectory_rewards =\
            [], [], [], []

    def load_weights(self, name):
        self.model.load_weights(name)

    def save_weights(self, name):
        self.model.save_weights(name)


