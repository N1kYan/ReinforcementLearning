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

tf_session = tf.Session()
K.set_session(tf_session)

env = gym.make("Pendulum-v0")
state_size = len(env.reset())
action_size = len(env.action_space.sample())
lr = 0.01
policy_gradient_lr = 0.01
hidden_neurons = state_size*10
replay_buffer_size = 200
N = 50  # Batch size
episodes = 10
gamma = 0.85
tau = 0.5

print("\nEnvironment {}".format(env.spec))
print("{} dimensional observation space".format(state_size))
print("{} dimensional action space".format(action_size))


def ddpg():

    # Initialize actor and critic networks
    critic = Critic(state_size=[state_size], action_size=[action_size], learningrate=lr,
                    hidden_neurons=hidden_neurons, session=tf_session)
    actor = Actor(state_size=[state_size], action_size=[action_size], learningrate=lr,
                  hidden_neurons=hidden_neurons, session=tf_session, action_high=env.action_space.high,
                  action_low=env.action_space.low)

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(buffer_number=replay_buffer_size)

    for e in range(episodes):

        cumulated_reward = 0

        # TODO: Random process for action exploration
        # Initial observation state
        state = env.reset()

        # Iteration 1...T
        while True:

            # Only render last episode
            if e == episodes-1:
                env.render()

            # Predict action by forwarding reshaped action through actor network
            action = actor.nn.predict(state.reshape((1, -1)))
            # Perform predicted action
            new_state, reward, done, info = env.step(action)
            cumulated_reward += reward
            # Store transition in replay buffer
            replay_buffer.add_observation(state, action, reward, new_state, done)
            # Sample random minibatch of observations from replay buffer
            sampled_states, sampled_actions, sampled_rewards, sampled_next_states,\
                sampled_dones = replay_buffer.random_batch(batch_size=N)
            # Predict target actions from sampled states
            target_actions = actor.nn_target.predict(x=sampled_states)
            # Predict target q values from predicted actions and sampled states
            target_Qs = critic.target_nn.predict(x=[sampled_states, target_actions])
            Y = sampled_rewards + gamma * target_Qs
            # Update critic by minimizing squared loss between Y and estimated q values from critic
            critic.nn.train_on_batch(x=[sampled_states, sampled_actions.reshape(-1, 1)], y=Y)

            # Predict actions from sampled states by actor
            predicted_actions = actor.nn.predict(x=sampled_states)


            # TODO: Actor gradient w.r.t. its parameters
            grad_mu = np.array(tf.gradients(ys=predicted_actions,
                                            xs=actor.nn.trainable_weights))
            # TODO: Critic gradient w.r.t. the actor's actions
            grad_Q = np.array(tf.gradients(ys=critic.nn.predict(x=[sampled_states, predicted_actions]),
                                           xs=predicted_actions))

            # TODO: Update actor network weights
            print(grad_mu)  # [None None None None]
            #prod = np.multiply(x1=grad_Q, x2=grad_mu)
            prod = np.dot(grad_Q, grad_mu)
            len_prod = len(prod)
            grad_J = np.sum(prod)/len_prod
            actor.nn.trainable_weights += policy_gradient_lr * grad_J

            # Update target network weights
            critic.train_target(tau=tau)
            actor.train_target(tau=tau)

            if done:
                break
        print("Episode {} Cumulated Reward {}".format(e, cumulated_reward))


if __name__ == "__main__":
    ddpg()

