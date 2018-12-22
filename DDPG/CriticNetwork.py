from keras.layers import Dense, Input, Activation, Add, Multiply, Lambda
from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf


class Critic(object):
    def __init__(self, state_size, action_size, learningrate, hidden_neurons, session):
        self.learning = learningrate
        self.state_space = state_size[0]
        self.action_space = action_size[0]
        self.hidden = hidden_neurons
        self.tensor_session = session


        self.state_input, self.action_input, self.nn, self.weights = self.create_network()
        self.target_state_input, self.target_action_input, self.target_nn, self.target_weights = self.create_network()

        self.critic_gradient = tf.gradients(self.nn.output, self.action_input)
        self.tensor_session.run(tf.initialize_all_variables())

    def create_network(self):
        state_input = Input(shape=(self.state_space,))
        action_input = Input(shape=(self.action_space,))
        s1 = Dense(self.hidden, activation='relu')(state_input)
        a1 = Dense(self.hidden, activation='relu')(action_input)

        merge = Add()([state_input, action_input])  # bring both inputs together could also work with merge()  # s1, a1?
        m1 = Dense(self.hidden, activation='relu')(merge)

        output = Dense(1, activation='relu')(m1)
        output = Lambda(lambda x: x*2)(output)

        nn = Model(input= [state_input, action_input], output=output)
        adam_optimizer = Adam(self.learning)
        nn.compile(loss='mse', optimizer=adam_optimizer)
        return state_input, action_input, nn, nn.trainable_weights

    def train(self, state, action):
        trained = self.tensor_session.run(self.critic_gradient, feed_dict={self.state_input:state, self.action_input:action})
        return trained[0]

    def train_target(self):
        # hard update: critic weights will be the target weights
        self.target_nn.set_weights(self.nn.get_weights())
