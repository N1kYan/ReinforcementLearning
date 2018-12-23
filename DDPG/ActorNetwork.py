from keras.layers import Dense, Input, Activation, Add, Multiply, Lambda
from keras.optimizers import Adam
from keras.models import Model, Sequential
import tensorflow as tf
import numpy as np
from keras import backend as K


class Actor(object):
    def __init__(self, state_size, action_size, learningrate, hidden_neurons, session, action_high, action_low):
        self.learning = learningrate
        self.state_space = state_size
        self.action_space = action_size[0]
        self.hidden = hidden_neurons
        self.tensor_session = session
        self.low = -1*action_low
        self.high = action_high

        self.input, self.nn, self.weights = self.create_network()
        self.input_target, self.nn_target, self.weights_target = self.create_network()

        self.merge_grad = tf.placeholder(tf.float32, [None, self.action_space])  # this will be the saving place for the deterministic policy gradient (theorem)
        self.actor_gradient = tf.gradients(self.nn.output, self.weights, -self.merge_grad)
        self.optimizing = tf.train.AdamOptimizer(self.learning).apply_gradients(zip(self.actor_gradient, self.weights))

        self.tensor_session.run(tf.initialize_all_variables())

    '''
    def activateHighLow(self, x):
        activated_x = K.tanh(x)
        #activated_x = tf.cond(x<0, tf.multiply(x, self.low), tf.multiply(x, self.high))
        activated_x = tf.multiply(x, 15)
        print(activated_x)
        #activated_x = activated_x[np.where(activated_x < 0)] * self.low
        #activated_x = activated_x[np.where(activated_x >= 0)] * self.high
        return activated_x
    '''

    def create_network(self):
        net_input = Input(shape=self.state_space)
        h1 = Dense(self.hidden, activation='relu')(net_input)
        #output = Dense(units=self.action_space, activation=self.activateHighLow)(h1)
        output = Dense(units=self.action_space, activation='tanh')(h1)
        #change to variable output (not hard coded)
        output = Lambda(lambda x: x*2)(output)

        nn = Model(input=net_input, output=output)
        adam_optimizer = Adam(self.learning)
        nn.compile(loss='mse', optimizer=adam_optimizer)
        return net_input, nn, nn.trainable_weights

    def train(self, net_input, action_gradients):
        self.tensor_session.run(self.optimizing, feed_dict={self.input: net_input, self.merge_grad:action_gradients})

    def train_target(self, tau):
        # weights = self.nn.get_weights()
        # hard target update, all target weights are the current actor model weights
        new_weights = tau * self.nn.get_weights() + (1 - tau) * self.target_nn.get_weights()
        self.nn_target.set_weights(new_weights)


