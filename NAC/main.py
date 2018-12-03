import gym
import numpy as np
import tensorflow as tf
import keras.backend as K
from quanser_robots import GentlyTerminating
from quanser_robots.double_pendulum.examples import metronom

from NaturalActorCritic import NaturalActorCritic
from ActorCritc import ActorCritic


def evaluate(env, ctrl):
    obs = env.reset()

    while True:
        env.render()
        if ctrl is None:
            act = env.action_space.sample()
        else:
            act = np.array(ctrl(obs))
        obs, rwd, done, info = env.step(act)

    env.close()


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
    # TODO: sess?
    sess = tf.Session()
    K.set_session(sess)
    nac = ActorCritic(env, sess)

    num_trials = 10000
    trial_len = 500

    cur_state = env.reset()
    action = env.action_space.sample()
    while True:
        env.render()
        cur_state = cur_state.reshape((1, env.observation_space.shape[0]))

        action = nac.act(cur_state)
        action = action.reshape((1, env.action_space.shape[0]))

        new_state, reward, done, _ = env.step(action)
        new_state = new_state.reshape((1, env.observation_space.shape[0]))

        nac.remember(cur_state, action, reward, new_state, done)
        nac.train()

        cur_state = new_state

if __name__ == "__main__":
    main()
