import gym
import numpy as np
from quanser_robots import GentlyTerminating
from quanser_robots.double_pendulum.examples import metronom
env = GentlyTerminating(gym.make('DoublePendulum-v0'))
"""
    The DoublePendulum-v0 environment:
    
    The state space is 6 dimensional:
     (x, theta1, theta2, x dot, theta1 dot, theta2 dot)
    Min: (-2, -pi, -30, -40) Max: (2, pi, 30, 40)
    
    The action space is 1 dimensional.
    Min -15 Max: 15
"""

ctrl = ...  # some function f: s -> a
obs = env.reset()


def evaluate(env, crtl=None):

    obs = env.reset()

    while True:
        env.render()
        if crtl is None:
            act = env.action_space.sample()
        else:
            act = np.array(crtl(obs))
        obs, rwd, done, info = env.step(act)

    env.close()

def main():
    #ctrl = metronom.MetronomCtrl()
    evaluate(env)

if __name__ == "__main__":
    main()