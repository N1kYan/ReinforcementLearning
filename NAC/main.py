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
        # Draw action from actor network
        print("Current state: {} Shape: {}".format(state, np.shape(state)))
        # Predict gives LIST of output VECTOR so we have to take [0][0] from it
        action = act.predict(state)[0][0]
        # TODO: actor network only outputting discrete values? Or values suiting the env.action_space
        print("Chosen action: {} -> {}".format(action, int(action)))
        print()
        # Perform action and observe next state and reward
        next_state, reward, done, info = env.step(int(action))


def initialize(env, sess):
    actor = ActorNetwork(env=env, sess=sess)
    actor_model = actor.create_actor_model()[1]
    critic = CriticNetwork(env=env, sess=sess)
    critic_model = critic.create_critic_model()[1]
    return actor_model, critic_model


def main():
    # env = GentlyTerminating(gym.make('DoublePendulum-v0'))
    env = gym.make('CartPole-v0')
    """
        The DoublePendulum-v0 environment:

        The state space is 6 dimensional:
         (x, theta1, theta2, x dot, theta1 dot, theta2 dot)
        Min: (-2, -pi, -30, -40) Max: (2, pi, 30, 40)

        The action space is 1 dimensional.
        Min -15 Max: 15

    """
    # env = gym.make("DoublePendulum-v0')
    # TODO: sess?
    sess = tf.Session()
    K.set_session(sess)
    actor, critic = initialize(env=env, sess=sess)
    nac_with_lstd(env=env, sess=sess, act=actor, crit=critic, epochs=10, phi=...,
                  delta=0.1, alpha=0.1, beta=0.1, epsilon=1e-2)


if __name__ == "__main__":
    main()
