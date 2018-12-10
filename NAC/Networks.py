import gym
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam

"""
    This file stores all the network we use to create our NAC algorithm.
    Mainly this is the actor network and the critic network.
"""

class ActorNetwork:
    """

    """
    def __init__(self, env, sess):
        """
        Initialize Actor Network
        :param env: the environment (e.g. DoublePendulum, CartPole, etc.)
        :param sess: the current tensorflow session
        """
        self.env = env
        self.sess = sess

    def create_actor_model(self):
        """
        Create the NAC actor model and return it together with the state Input
        objects.
        :return: state Input object , the Model object
        """
        # Netowrk structure with 3 hidden layers
        # state_input = Input(shape=self.env.observation_space.shape)
        state_input = Input(shape=(4,))
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)

        # output = Dense(self.env.action_space.shape[0], activation='relu')(h3)
        #output = Dense(1, activation='relu')(h3)
        output = Dense(1)(h3)
        model = Model(input=state_input, output=output)
        adam = Adam(lr=0.001)
        model.compile(loss='mse', optimizer=adam)

        return state_input, model


class CriticNetwork:
    def __init__(self, env, sess):
        """
        Initialize CriticNetwork.
        :param env: the environment (e.g. DoublePendulum, CartPole, etc.)
        :param sess: the current tensorflow session
        """
        self.env = env
        self.sess = sess

    def reshape_action_space(self):
        """
        Return the shape of the action shape, correctly encapsulated in an
        1 dim array with 2 fields. This 'reshaping' is needed for Discrete gym
        action spaces. These discrete gym action spaces have no proper shape
        parameter, so we build one ourselves.
        :return: shape of our action space
        """
        if self.env.action_space.n:
            # env.action_space.n is only defined in discrete action spaces
            return (self.env.action_space.n,1)
        else:
            return self.env.action_space.shape

    def create_critic_model(self):
        """
        Create the NAC critic model and return it together with the state/
        action Input objects.
        :return: state Input object , action Input object , the Model object
        """

        dim_actions = self.reshape_action_space()

        state_input = Input(shape=(4,))
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=(1,))
        action_h1 = Dense(48)(action_input)

        # We are adding our 48 neurons of our state layer to our 48 neurons
        # of our action layer.
        # What effect does this have ???
        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model = Model(input=[state_input, action_input], output=output)

        adam = Adam(lr=0.001)
        model.compile(loss='mse', optimizer=adam)
        return state_input, action_input, model
