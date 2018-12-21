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
    action_space = env.action_space.shape
    state_space = env.observation_space.shape
    hidden_layer = 100

    buffer_size = 5000
    batch_size = 32
    learning_rate = 0.001

    tensor = tf.Session()
    K.set_session(tensor)

    actor = Actor(state_space, action_space, learning_rate, hidden_layer, tensor)
    critic = Critic(state_space, action_space, learning_rate, hidden_layer, tensor)
    replay = ReplayBuffer(buffer_size)

    episodes = 100
    gamma = 0.99
    step_size = 200
    epsilon = 0.1

    step = 0

    for epi in range(episodes):
        state = env.reset()

        total_reward = 0

        for step in range(step_size):
            if epi == episodes-1:
                env.render()
            loss = 0
            if random.random() > epsilon:
                #exploitation = use knowledge
                #action = actor.nn.predict(state)
                action = actor.nn.predict(state.reshape(1, state.shape[0]))

            else:
                #exploration = use random sample of the action space
                action = np.random.uniform(env.action_space.low, env.action_space.high, size=(1, action_space[0]))

            #take the action and add it to the memory
            state_follows, reward, done, info = env.step(action)
            replay.add_observation(state, action, reward, state_follows, [done])  # what is time? -> changed to dones

            #batch update
            if len(replay.ReplayBuffer) < 2:
                print("Replay buffer smaller than batch size")
                state = state_follows
                continue
            states, actions, rewards, next_states, dones = replay.random_batch(batch_size)
            actions = np.concatenate(actions)
            #q = r+gamma*next_q if not done else q = r

            target_q = critic.nn.predict([next_states, actor.nn.predict(next_states)])
            q = rewards[:]

            q[np.where(dones==False)] += (gamma * target_q[np.where(dones==False)])
            # TODO: Actor and critic updates
            loss += critic.nn.train_on_batch([states, actions], q)
            pred_action = actor.nn.predict(states)
            gradients = critic.train(states, pred_action)
            actor.train(states, gradients)
            actor.train_target()
            critic.train_target()

            state = state_follows
            total_reward += reward

            step +=1
            if done:
                break
        print("Episode", epi, "Step", step, "Action", action, "Reward", total_reward, "Loss", loss, "Epsilon", epsilon)


if __name__ == "__main__":
    environment = gym.make('Qube-v0')
    ddpg(environment)
