import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import quanser_robots
import sys
#from quanser_robots import GentlyTerminating
#from quanser_robots.double_pendulum.examples import metronom

from NaturalActorCritic import *
from ActorCritc import *
from Networks import *

from collections import deque


# What is ctrl ???
def evaluate(env, ctrl, episodes):
    obs = env.reset()

    for e in range(episodes):
        while True:
            env.render()
            if ctrl is None:
                action = env.action_space.sample()
            else:
                action = np.array(ctrl(obs))
            obs, rwd, done, info = env.step(action)

            if done:
                break


# TODO
def _critic_evaluation(memory):
    for sample in memory:
        cur_state, action, reward, new_state, done = sample
        if not done:
            target_action = self.target_actor_model.predict(new_state)
            future_reward = self.target_critic_model.predict(
                [new_state, target_action])[0][0]
            reward += self.gamma * future_reward
        self.critic_model.fit([cur_state, action], reward, verbose=0)


# TODO
def _actor_update():
    pass


def episodic_nac(env, sess, updates, epochs, actor, critic, alpha, gamma):
    # Parameters of actor network
    # theta = act.trainable_weights
    # Gradient of actor network
    # grad = act.weight

    for u in range(updates):
        memory = deque()

        for e in range(epochs):
            state = env.reset()
            state = state.reshape((1, 4))


            t = 0
            while True:
                t += 1
                #TODO: Sotfmax?
                action = np.argmax(actor.predict(state)[0])
                next_state, reward, done, _ = env.step(action)

                # Save all the states of the episode
                memory.append([state, action, reward, next_state, done])

                state = np.copy(next_state).reshape((1, 4))

                if done:
                    print("Epoch {} done after {} timesteps".format(e, t))
                    break

        _critic_evaluation()
        _actor_update()



def nac_with_lstd(env, sess, act, crit, epochs, phi, delta, alpha, beta, epsilon):
    # Draw initial state and reshape for network input
    # state = draw_inital_state()
    state = env.reset()
    state = np.asarray(state).reshape((1, 4))
    # Initialize parameters
    A = 0
    b = 0
    z = 0

    for t in range(epochs):
        env.render()
        # Draw action from actor network
        print("Current state: {} Shape: {}".format(state, np.shape(state)))
        # Predict gives LIST of output VECTOR so we have to take [0][0] from it
        action = np.argmax(act.predict(state)[0])
        # TODO: actor network only outputting discrete values? Or values suiting the env.action_space
        print("Chosen action: ", action)
        print()
        # Perform action and observe next state and reward
        next_state, reward, done, info = env.step(int(action))
        state = np.copy(next_state).reshape((1, 4))

        if done:
            print("Epoch finished after {} timesteps".format(t))
            break


def initialize(env, sess):
    """
    Return the actor and the critic model of our NAC algorithm.
    :param env: the environment (e.g. DoublePendulum, CartPole, etc.)
    :param sess: the current tensorflow session
    :return: actor model, critic model
    """
    actor = ActorNetwork(env=env, sess=sess)
    _, actor_model = actor.create_actor_model()
    critic = CriticNetwork(env=env, sess=sess)
    _, _, critic_model = critic.create_critic_model()
    return actor_model, critic_model


def main():
    # env = GentlyTerminating(gym.make('DoublePendulum-v0'))
    """
        The DoublePendulum-v0 environment:

        The state space is 6 dimensional:
         (x, theta1, theta2, x dot, theta1 dot, theta2 dot)
        Min: (-2, -pi, -30, -40) Max: (2, pi, 30, 40)

        The action space is 1 dimensional.
        Min -15 Max: 15

    """
    # env = gym.make("DoublePendulum-v0')
    env = gym.make('CartPole-v0')
    sess = tf.Session()
    K.set_session(sess)

    # Get the actor and critic model of our algorithm
    actor, critic = initialize(env=env, sess=sess)

    episodic_nac(env=env, sess=sess, updates=1, epochs=100, actor=actor, critic=critic, alpha=0.5, gamma=0.8)

    #nac_with_lstd(env=env, sess=sess, act=actor, crit=critic, epochs=1000, phi=...,
    #              delta=0.1, alpha=0.1, beta=0.1, epsilon=1e-2)

    # sess.close()


if __name__ == "__main__":
    main()
