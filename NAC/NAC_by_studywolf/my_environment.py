import random
import numpy as np
import sys
import gym
import quanser_robots
from quanser_robots import GentlyTerminating


class MyEnvironment(gym.Space):
    def __init__(self, env_name, num_of_actions, time_steps):
        gym.Space.__init__(self, (), np.float)

        self.name = env_name
        self.num_of_actions = num_of_actions
        self.time_steps = time_steps  # per trajectory

        # self.env = GentlyTerminating(gym.make(env_name))  # doesnt work
        self.env = gym.make(env_name)
        self.env = gym.wrappers.Monitor(
            env=self.env,
            directory=env_name + '/',
            force=True,
            video_callable=False)

        # OBSERVATION SPACE
        self.observation_space = self.env.observation_space
        self.observation_space.high = self.env.observation_space.high
        self.observation_space.low = self.env.observation_space.low

        print("\tObservation space high: {}".format(self.observation_space.high))
        print("\tObservation space low : {}".format(self.observation_space.low))

        # ACTION SPACE
        if type(self.env.action_space) is gym.spaces.discrete.Discrete:
            assert num_of_actions is None
            self.action_space = np.arange(self.env.action_space.n)
        elif type(self.env.action_space) is gym.spaces.box.Box \
                or type(self.env.action_space) is quanser_robots.common.LabeledBox:
            assert num_of_actions is not None
            self.action_space = np.linspace(self.env.action_space.low,
                                            self.env.action_space.high,
                                            self.num_of_actions)
        else:
            raise ValueError("Env Action Space should be of type Discrete "
                             "or Box, but is of Type {}.".format(type(self.env.action_space)))

        self.env.action_space = self.action_space
        self.action_space_high = np.max(self.action_space)
        self.action_space_low = np.min(self.action_space)

        print("\tAction space: {}".format(self.action_space))

    def step(self, action):
        return self.env.step(action)

    def sample_env_action(self):
        return self.env.action_space.sample()

    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()

    def render(self):
        self.env.render()

    def action_space_contains(self, x):
        """
        :param x: action space values (as array or similar)
        :return: index of the given actions (as list)
        """
        indices = []
        for i in x:
            indices.append(np.where(self.action_space == i)[0][0])
        return indices

    def action_space_sample(self):
        return random.choice(self.action_space)
