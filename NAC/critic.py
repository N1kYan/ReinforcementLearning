import tensorflow as tf
import numpy as np


class Critic:

    def __init__(self, env):
        self.state_input, self.true_vf_input, self.output, \
            self.optimizer, self.loss = self.create_value_net(env)

    def update(self, sess, batch_states, batch_discounted_returns):
        returns_vector = np.expand_dims(batch_discounted_returns, axis=1)
        sess.run(self.optimizer,
                 feed_dict={self.state_input: batch_states,
                            self.true_vf_input: returns_vector})

    def estimate(self, sess, observation):
        # The estimated return the critic expects the state to have
        observation_vector = np.expand_dims(observation, axis=0)
        state_value = sess.run(
            self.output,
            feed_dict={self.state_input: observation_vector}
        )[0][0]

        return state_value

    @staticmethod
    def create_value_net(env):
        """
        Function approximation of the value function for states in our environment.
        We use a neural network with the following structure:
            Input Layer: number of nodes = dim of state, fully connected.
            Hidden Layer: fully connected, ReLu activation.
            Output Layer: 1 node.
        Finally, we use an AdamOptimizer to train our NN by Gradient Descent.

        :param env: the environment we are working with
        :return:
            estimated value of our neural network,
            placeholder variable to input the state into the network,
            placeholder variable to input the true value function value for the
                above state,
            the adam optimizer object,
            the loss between the true value and the estimated value for the state
        """
        with tf.variable_scope("critic"):
            # Get the state size to get the number of input nodes
            state_size = env.observation_space.shape[0]

            # Input layer, hidden dense layer,
            # bias b1 & ReLu activation
            state_input = tf.placeholder("float", [None, state_size])
            w1 = tf.get_variable("w1", [state_size, env.hidden_layer_critic])
            b1 = tf.get_variable("b1", [env.hidden_layer_critic])
            h1 = tf.nn.relu(tf.matmul(state_input, w1) + b1)

            # Output times 2nd weights plus 2nd bias
            w2 = tf.get_variable("w2", [env.hidden_layer_critic, 1])
            b2 = tf.get_variable("b2", [1])
            output = tf.matmul(h1, w2) + b2

            # During runtime this value will hold the true value
            # (discounted return) of the value function which we will use to
            # adjust our NN accordingly.
            true_vf_input = tf.placeholder("float", [None, 1])

            # Minimize the difference between predicted and actual output
            diffs = output - true_vf_input
            loss = tf.nn.l2_loss(diffs)  # sum (diffs ** 2) / 2

            # Computes the gradients of the network and applies them again so
            # the 'loss' value will be minimized. This is done via the Adam
            # algorithm.
            optimizer = tf.train.\
                AdamOptimizer(env.learning_rate_critic).minimize(loss)

            return state_input, true_vf_input, output, optimizer, loss
