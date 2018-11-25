import tensorflow as tf
import numpy as np
import gym
import quanser_robots
import tensorflow as tf
from keras.layers import Dense, Input, Activation, Add, Multiply
from keras.optimizers import Adam
from keras.models import Model
import keras.backend as K

import random
from collections import deque


class Replay(object):
    def __init__(self, buffer_number):
        self.size = buffer_number
        self.ReplayBuffer = deque()

    def add_observation(self, state, action, reward, next_state, time):
        self.ReplayBuffer.append([state, action, reward, next_state, time])
        if len(self.ReplayBuffer) > self.size:
            self.ReplayBuffer.popleft()

    def random_batch(self, batch_size):
        element_number = len(self.ReplayBuffer)
        if element_number > batch_size:
            batch_size = element_number
        expectations = random.sample(self.ReplayBuffer, k=element_number)
        states = list(zip(*expectations))[0]
        actions = list(zip(*expectations))[1]
        rewards = list(zip(*expectations))[2]
        next_states = list(zip(*expectations))[3]
        times = list(zip(*expectations))[4]
        return states, actions, rewards, next_states, times

'''
class Actor(nn.Module):
    def __init__(self, state_size, action_size, learningrate, batch_size, hidden_neurons, action_limit):
        self.learning = learningrate
        self.state_space = state_size
        self.action_space = action_size
        self.replay_buffer = Replay(5000)
        #self.tau = 0.1

        self.action_limit = action_limit

        self.input = nn.Linear(self.state_space, hidden_neurons)
        self.input.weight.data.uniform_(-0.05, 0.05)

        self.hidden = nn.Linear(hidden_neurons, self.action_space)
        self.hidden.weight.data.uniform_(-0.05,0.05)

    def forward(self, state):
        f1 = F.relu(self.input(state))
        f2 = F.relu(self.hidden(f1))
        out = F.tanh(self.output(f2))

        out = out*self.action_limit
        return out


class Critic(nn.Module):
    def __init__(self, state_size, action_size, learningrate, batch_size, hidden_neurons, action_limit):
        self.learning = learningrate
        self.state_space = state_size
        self.action_space = action_size
        self.replay_buffer = Replay(5000)
        #self.tau = 0.1

        self.action_limit = action_limit

        self.input = nn.Linear(self.state_space, hidden_neurons)
        self.input.weight.data.uniform_(-0.05, 0.05)

        self.hidden = nn.Linear(hidden_neurons+ self.action_space, hidden_neurons)
        self.hidden.weight.data.uniform_(-0.05,0.05)

        self.out = nn.Linear(hidden_neurons, 1)
        self.out.weight.data.uniform_(-0.05,0.05)


    def forward(self, state, action):
        f1 = F.relu(self.input(state))

        x = torch.cat((f1, action),1)
        x = self.hidden(x)
        out = F.relu(x)

        out = self.out(out)
        return out
'''


class Actor(object):
    def __init__(self, state_size, action_size, learningrate, hidden_neurons, action_dim, session):
        self.learning = learningrate
        self.state_space = state_size
        self.action_space = action_size
        self.hidden = hidden_neurons
        self.action_dim = action_dim
        self.input = None #will hold the input layer
        self.output = None #will hold the output layer
        self.nn = None #will hold the model of the nn
        self.tensor_session = session

    def create_network(self):
        self.input = Input(self.state_space)
        h1 = Dense(self.hidden, activation='relu')(self.input)
        output = Dense(self.action_space, activation='tanh')(h1)
        self.output = Multiply(self.action_dim)(output)
        self.nn = Model(input=self.input, output=self.output)
        adam_optimizer = Adam(self.learning)
        self.nn.compile(loss='mse', optimizer=adam_optimizer)
        return self.input, self.nn, self.nn

    def train(self, optimizer, input, action_gradient, merge):
        merged_grad = merge
        self.tensor_session.run(optimizer, feed_dict={self.input: input, merged_grad:action_gradient})

    def predict(self, input):
        self.tensor_session.run(self.output, feed_dict={self.input:input})

    def update_target(self, actor, critic):
        return

class Critic(object):
    def __init__(self, state_size, action_size, learningrate, hidden_neurons, session):
        self.learning = learningrate
        self.state_space = state_size
        self.action_space = action_size
        self.hidden = hidden_neurons
        self.tensor_session = session
        self.state_input = None
        self.action_input = None
        self.output = None
        self.nn = None

    def create_network(self):
        self.state_input = Input(self.state_space)
        self.action_input = Input(self.action_space)
        s1 = Dense(self.hidden, activation='relu')(self.state_input)
        a1 = Dense(self.hidden, activation='relu')(self.action_input)

        merge = Add()(self.state_input, self.action_input)
        m1 = Dense(self.hidden, activation='relu')(merge)

        self.output = Dense(1, activation='relu')(m1)

        self.nn = Model(input= [self.state_input, self.action_input], output=self.output)
        adam_optimizer = Adam(self.learning)
        self.nn.compile(loss='mse', optimizer=adam_optimizer)
        return self.state_input, self.action_input, self.nn, self.nn

    def train(self, optimizer, merge, state, action, pred_q):
        self.tensor_session.run([self.output, optimizer], feed_dict={self.state_input:state, self.action_input:action, merge:pred_q})

    def predict(self, state, action):
        self.tensor_session.run(self.output, feed_dict={self.state_input: state, self.action_input:action})


class ActorCriticTargets(object):
    def __init__(self, state_space, action_space, learningrate, hidden_neurons):
        self.replay = Replay(5000)
        self.tau = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0
        self.eps_dec = 0.9

        self.tensor = tf.Session()
        K.set_session(self.tensor)

        self.actor = Actor(state_space, action_space, learningrate, hidden_neurons, self.session)
        self.actor_input_dim, self.actor_model, self.actor_target_model = self.actor.create_network()
        self.actor_weights = self.actor_model.trainable_weights
        self.merge_grad = tf.placeholder(tf.float32, [None, action_space]) # this will be the saving place for the deterministic policy gradient (theorem)
        self.actor_gradient = tf.gradients(self.actor_model.output, self.actor_weights, -self.merge_grad)

        self.optimizing = tf.train.AdamOptimizer(learningrate).apply_gradients(zip(self.actor_gradient, self.actor_weights))

        self.critic = Critic(state_space, action_space, learningrate, hidden_neurons, self.session)
        self.state_input, self.action_input, self.critic_model, self.critic_target_model = self.critic.create_network()
        self.critic_gradient = tf.gradients(self.critic_model.output, self.action_input)

        self.tensor.run(tf.initializ_all_variables())

    def train(self):
        batch_size = 64
        if self.replay.size<batch_size:
            return
        sample = self.replay.random_batch(self.replay, batch_size)
        self.critic.train(self.critic_model)
        self.actor.train(self.actor_model)

env = gym.make('Qube-v0')
print(env.action_space.shape[0])