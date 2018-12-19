from keras.layers import Dense, Input, Activation, Add, Multiply
from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf


class Actor(object):
    def __init__(self, state_size, action_size, learningrate, hidden_neurons, session):
        self.learning = learningrate
        self.state_space = state_size
        self.action_space = action_size
        self.hidden = hidden_neurons
        self.tensor_session = session

        self.input, self.nn, self.weights = self.create_network()
        self.input_target, self.nn_target, self.weights_target = self.create_network()

        self.merge_grad = tf.placeholder(tf.float32, [None, self.action_space])  # this will be the saving place for the deterministic policy gradient (theorem)
        self.actor_gradient = tf.gradients(self.nn.output, self.weights, -self.merge_grad)
        self.optimizing = tf.train.AdamOptimizer(self.learning).apply_gradients(zip(self.actor_gradient, self.weights))

        self.tensor_session.run(tf.initialize_all_variables())

    def create_network(self):
        net_input = Input(self.state_space)
        h1 = Dense(self.hidden, activation='relu')(net_input)
        output = Dense(self.action_space, activation='tanh')(h1)
        nn = Model(input=net_input, output=output)
        adam_optimizer = Adam(self.learning)
        nn.compile(loss='mse', optimizer=adam_optimizer)
        return net_input, nn, nn.trainable_weights

    def train(self, net_input, action_gradients):
        self.tensor_session.run(self.optimizing, feed_dict={self.input: net_input, self.merge_grad:action_gradients})

    def train_target(self):
        weights = self.weights
        # hard target update, all target weights are the current actor model weights
        self.nn_target.set_weights(weights)
