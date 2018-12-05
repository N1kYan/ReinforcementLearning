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


def initialize(env, sess):
    actor = ActorNetwork(env=env, sess=sess)
    actor.create_actor_model()
    critic = CriticNetwork(env=env, sess=sess)
    critic.create_critic_model()

def main():
    #env = GentlyTerminating(gym.make('DoublePendulum-v0'))
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
    initialize(env=env, sess=sess)



if __name__ == "__main__":
    main()
