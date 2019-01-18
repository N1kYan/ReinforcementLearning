
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K
import random

import tensorflow as tf


class Actor:
    def __init__(self, env, sess, det_action_select, disc_actions=None):
        self.env = env
        self.sess = sess
        self.det_action_select = det_action_select
        self.disc_actions = disc_actions
        if det_action_select and disc_actions is None:
            raise ValueError("Please specify disc_actions for det actions")
        self.actor_state_input, self.actor_model = self.create_actor_model()

    def create_actor_model(self):
        state_input = Input(shape=self.env.observation_space.shape)
        h1 = Dense(100, activation='relu')(state_input)
        output = Dense(self.env.action_space.shape[0], activation='tanh')(h1)

        model = Model(inputs=state_input, outputs=output)
        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

        # TODO: Need last layer to sum to one
        # TODO: Do we take Keras or low level tensorflow

    def predict(self, state):
        """
        Return a one-hot array where we chose the action which we chose
        according to our policy. An action which has a high probability in
        the current state is chosen far more often than one with low
        probability.

        :param state: The state where we want to take the action from
        :return: one-hot array with size len(actions)
        """
        probs = self.actor_model.predict(state)

        # Check which action to take
        probs_sum = 0
        action = None
        for i in range(self.disc_actions.length()):
            probs_sum += probs[0][i] # TODO: is this access correct?
            rnd = random.uniform(0, 1)
            if rnd < probs_sum:
                action = i

        # Make one-hot action array
        action_array = np.zeros(self.disc_actions.length())
        action_array[action] = 1

        return action_array

    def get_gradients(self, input_state):
        """
        We need the gradients of a specific input of the actor network to
        successfully calculate the weights of the critic network.

        :param input_state: the state which we input to calculate the gradients
        :return: the gradients of the actor network for 'input_state'
        """
        grads = self.sess.run(tf.gradients(self.actor_model.output,
                                           self.actor_model.trainable_weights),
                              feed_dict={self.actor_state_input:  input_state}
                              )[0]

        return grads

    def update(self, w_new, alpha):
        # TODO: w_new probably hasn't the same dimension as theta
        old_theta = self.actor_model.get_weights()
        new_theta = old_theta + alpha * w_new
        self.actor_model.set_weights(new_theta)




