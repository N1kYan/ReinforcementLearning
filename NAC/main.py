import gym
import numpy as np
import tensorflow as tf
import keras.backend as K
from quanser_robots import GentlyTerminating
from quanser_robots.double_pendulum.examples import metronom

from NaturalActorCritic import *
from ActorCritc import *
from Networks import *


def evaluate(env, ctrl):
    obs = env.reset()

    while True:
        env.render()
        if ctrl is None:
            act = env.action_space.sample()
        else:
            act = np.array(ctrl(obs))
        obs, rwd, done, info = env.step(act)


# TODO
def _critic_evaluation():
    None


# TODO
def _actor_update():
    None

def episodic_nac(env, sess, epochs, act, crit, alpha, gamma):
    # Parameters of actor network
    # theta = act.trainable_weights
    # Gradient of actor network
    # grad = act.weight

    for u in range(env.action_space.n):
        for e in range(epochs):
            state = env.reset().reshape((1, 4))
            t = 0
            while True:
                #TODO: Sotfmax?
                action = np.argmax(act.predict(state)[0])
                next_state, reward, done, infos = env.step(action)
                state = np.copy(next_state).reshape((1, 4))
                if done:
                    print("Epoch {} done after {} timesteps".format(e, t))
                    break
                t += 1

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
    actor_model = actor.create_actor_model()[1]
    critic = CriticNetwork(env=env, sess=sess)
    critic_model = critic.create_critic_model()[1]
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

    # TODO: Session
    # Create tensorflow session and set Keras session
    sess = tf.Session()
    K.set_session(sess)

    # Get the actor and critic model of our algorithm
    actor, critic = initialize(env=env, sess=sess)

    episodic_nac(env=env, sess=sess, epochs=100, act=actor, crit=critic, alpha=0.5, gamma=0.8)

    #nac_with_lstd(env=env, sess=sess, act=actor, crit=critic, epochs=1000, phi=...,
    #              delta=0.1, alpha=0.1, beta=0.1, epsilon=1e-2)


if __name__ == "__main__":
    main()
