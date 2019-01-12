
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K

import tensorflow as tf


class Critic:
    def __init__(self, env, sess):
        self.env = env

        self.critic_state_input, self.critic_action_input, \
        self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output,
                                         self.critic_action_input)  # where we calcaulte de/dC for feeding above

    def create_critic_model(self):
        state_input = Input(shape=self.env.observation_space.shape)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=self.env.action_space.shape)
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model = Model(inputs=[state_input, action_input], outputs=output)

        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    def predict(self, state, action):
        """Return future reward"""
        return self.critic_model.predict([state, action])[0][0]

    def get_weights(self):
        self.critic_model.get_weights()

    def set_weights(self, weights):
        self.critic_model.set_weights(weights)
