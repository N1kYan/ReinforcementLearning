import tensorflow as tf
import numpy as np
import gym
import quanser_robots

import random
from collections import deque

from ActorNetwork import Actor
from CriticNetwork import Critic
from ReplayBuffer import ReplayBuffer
from keras import backend as K


def ddpg(env):
    action_space = env.action_space.shape[0]
    state_space = env.observation_space.shape[0]
    hidden_layer = 100

    buffer_size = 5000
    batch_size = 1000
    learning_rate = 0.001

    tensor = tf.Session()
    K.set_session(tensor)

    actor = Actor(state_space, action_space, learning_rate, tensor, session=tensor)
    critic = Critic(state_space, action_space, learning_rate, tensor, session=tensor)
    replay = ReplayBuffer(buffer_size)

    episodes = 200
    step_size = 75
    epsilon = 0.1

    for epi in range(episodes):
        state = env.reset()

        for step in range(step_size):
            loss = 0
            if random.random() > epsilon:
                #exploitation = use knowledge
                action = actor.nn.predict(state.reshape(1, state.shape[0]))
            else:
                #exploration = use random sample of the action space
                action = np.random.uniform(env.action_space.low, env.action_space.high, size=(1, action_space))

            #take the action and add it to the memory
            state_follows, reward, done, info = env.step(action)
            replay.add_observation(state, action, reward, state_follows, done)  # what is time? -> changed to dones

            #batch update
            '''if len(replay.ReplayBuffer) < batch_size:
                print("Replay buffer smaller than batch size")
                continue'''
            states, actions, rewards, next_states, dones = replay.random_batch(batch_size)

            state = state_follows

            #q = r+gamma*next_q if not done else q = r
            target_q = critic.nn.predict([next_states, actor.nn.predict(next_states)])

            # TODO: Actor and critic updates

if __name__ == "__main__":
    environment = gym.make('Qube-v0')
    ddpg(environment)
