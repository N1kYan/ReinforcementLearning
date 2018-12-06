import gym
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam


class ActorNetwork:
    def __init__(self, env, sess):
        self.env = env
        self.sess = sess

    def create_actor_model(self):
        # Netowrk structure with 3 hidden layers
        # state_input = Input(shape=self.env.observation_space.shape)
        state_input = Input(shape=(4,))
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)

        # output = Dense(self.env.action_space.shape[0], activation='relu')(h3)
        #output = Dense(1, activation='relu')(h3)
        output = Dense(1) (h3)
        model = Model(input=state_input, output=output)
        adam = Adam(lr=0.001)
        model.compile(loss='mse', optimizer=adam)

        return state_input, model


class CriticNetwork:
    def __init__(self, env, sess):
        self.env = env
        self.sess = sess

    # This 'reshaping' is needed for Discrete gym action spaces 
    def reshape_action_space(self):
        if self.env.action_space.n:
            return (self.env.action_space.n,1)
        else:
            return self.env.action_space.shape

    def create_critic_model(self):

        dim_actions = self.reshape_action_space()

        state_input = Input(shape=self.env.observation_space.shape)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=dim_actions)
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model = Model(input=[state_input, action_input], output=output)

        adam = Adam(lr=0.001)
        model.compile(loss='mse', optimizer=adam)
        return state_input, action_input, model
