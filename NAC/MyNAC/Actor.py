
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K

import tensorflow as tf

class Actor:
    def __init__(self, env, sess):
        self.env = env
        self.sess = sess
        self.actor_state_input, self.actor_model = self.create_actor_model()

    def create_actor_model(self):
        state_input = Input(shape=self.env.observation_space.shape)
        h1 = Dense(100, activation='relu')(state_input)
        output = Dense(self.env.action_space.shape[0], activation='tanh')(h1)

        model = Model(inputs=state_input, outputs=output)
        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def predict(self, state):
        self.actor_model.predict(state)

    def get_gradients(self, input_state):
        """
        We need the gradients of a specific input of the actor network to
        successfully calculate the weights of the critic network.

        :param input_state: the state which we input to calculate the gradients
        :return: the gradients of the actor network for 'input_state'
        """
        grads = self.sess.run(tf.gradients(self.actor_model.output,
                                           self.actor_state_input),
                              feed_dict={self.actor_state_input:  input_state}
                              )[0]

        return grads

    def update(self, w_new, alpha):
        # TODO: w_new probably hasn't the same dimension as theta
        old_theta = self.actor_model.get_weights()
        new_theta = old_theta + alpha * w_new
        self.actor_model.set_weights(new_theta)




