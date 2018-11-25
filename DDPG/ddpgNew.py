import tensorflow as tf
import numpy as np
import gym
import quanser_robots

import random
from collections import deque

from ReplayBuffer import ReplayBuffer
from Networks import Actor, Critic
#from Networks import ActorTarget, CriticTarget

''

class ActorCriticTargets(object):
    def __init__(self, state_space, action_space, learningrate, hidden_neurons):
        self.replay = ReplayBuffer(5000)
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
            print("Replay buffer smaller than batch size")
            return
        sample = self.replay.random_batch(self.replay, batch_size)
        self.critic.train(self.critic_model)
        self.actor.train(self.actor_model)

env = gym.make('Qube-v0')
print(env.action_space.shape[0])