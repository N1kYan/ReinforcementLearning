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

    buffer_size = 1000
    batch_size = 500
    learning_rate = 0.001

    tensor = tf.Session()
    K.set_session(tensor)

    actor = Actor(state_space, action_space, learning_rate, hidden_layer, tensor, env.action_space.high, env.action_space.low)
    critic = Critic(state_space, action_space, learning_rate, hidden_layer, tensor)
    replay = ReplayBuffer(buffer_size)

    episodes = 50
    gamma = 0.95
    step_size = 200
    epsilon = 0.05
    epsilon_decay = 0.9995

    for epi in range(episodes):
        state = env.reset()

        total_reward = 0

        decaying_epsilon = 0.99

        for step in range(step_size):

            # Only render last episode to check learning
            if epi == episodes-1:
                env.render()

            loss = 0

            # Epsilon greedy policy for action selection
            if random.random() > decaying_epsilon:
                # Exploitation = use knowledge
                # action = actor.nn.predict(state)
                action = actor.nn.predict(state.reshape(1, state.shape[0]))

            else:
                # Exploration = use random sample of the action space
                action = np.random.uniform(env.action_space.low, env.action_space.high, size=(1, action_space[0]))
                # action = env.action_space.sample()

            # Take the action and add it to the replay buffer
            state_follows, reward, done, info = env.step(action)
            state_follows = np.concatenate(state_follows)
            replay.add_observation(state, action, reward, state_follows, [done])  # what is time? -> changed to dones

            # Batch update
            if len(replay.ReplayBuffer) < 2:  # < batch_size?
                print("Replay buffer smaller than batch size")
                state = state_follows
                continue

            states, actions, rewards, next_states, dones = replay.random_batch(batch_size)
            actions = np.concatenate(actions)
            #q = r+gamma*next_q if not done else q = r

            # Q-function target estimation from actor target network
            target_q = critic.target_nn.predict([next_states, actor.nn_target.predict(next_states)])
            q = rewards[:]
            # yi = ri + gamma * Q-target
            q[np.where(dones == False)] += (gamma * target_q[np.where(dones == False)])

            # TODO: Actor and critic updates
            # Update critic network weights
            loss += critic.nn.train_on_batch([states, actions], q)

            # Update actor network weights
            pred_actions = actor.nn.predict(states)  # We also need the actor gradient w.r.t. its parameters
            gradients = critic.train(states, pred_actions)  # Does this give us the gradient w.r.t. the actions?
            actor.train(states, gradients)

            # Update target networks weights
            actor.train_target(tau=0.25)
            critic.train_target(tau=0.25)

            state = state_follows
            total_reward += reward

            step += 1
            if done:
                break
        print("Episode", epi, "Step", step, "Action", action, "Reward", total_reward, "Loss", loss, "Epsilon", epsilon)
        decaying_epsilon *= epsilon_decay


if __name__ == "__main__":

    environment = gym.make('Pendulum-v0')
    #environment = gym.make('Qube-v0')
    ddpg(environment)
