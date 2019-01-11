
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K

import tensorflow as tf

class Actor:
    def __init__(self, env):
        self.env = env
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




