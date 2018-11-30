import gym
import numpy as np
from quanser_robots import GentlyTerminating
from quanser_robots.double_pendulum.examples import metronom
env = GentlyTerminating(gym.make('DoublePendulum-v0'))
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
    ctrl = metronom.MetronomCtrl()
    evaluate(env, ctrl)

if __name__ == "__main__":
    main()