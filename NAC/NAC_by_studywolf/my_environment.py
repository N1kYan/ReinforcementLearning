import random
import numpy as np
import sys
import gym
import quanser_robots
from quanser_robots import GentlyTerminating
import warnings
import datetime
import os

class MyEnvironment(gym.Space):
    def __init__(self, env_details):
        gym.Space.__init__(self, (), np.float)

        self.name = env_details[0]
        self.time_steps = env_details[3]  # between weight updates
        self.discount_factor = env_details[5]

        # -------------- CREATE FOLDER FOR SAVING FILES --------------------- #

        # Get current time
        save_time = datetime.datetime.now()

        self.save_folder = "{}/{}/{}-{}-{}_{}-{}-{}" \
            .format('data', self.name, save_time.year, save_time.month,
                    save_time.day, save_time.hour, save_time.minute,
                    save_time.second)

        # Create folder from current time
        try:
            os.makedirs(self.save_folder)
        except FileExistsError:
            pass

        # ------------------  CREATE GYM ENVIRONMENT ------------------------ #

        # Ignoring PkgResourcesDeprecationWarning: Parameters deprecated.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.env = gym.make(self.name)
            # self.env = GentlyTerminating(gym.make(env_name))  # doesnt work

        self.env = gym.wrappers.Monitor(
            env=self.env,
            directory=self.save_folder,
            force=True,
            video_callable=False)

        # OBSERVATION SPACE
        self.observation_space = self.env.observation_space
        self.observation_space.high = self.env.observation_space.high
        self.observation_space.low = self.env.observation_space.low

        print("\tObservation space high: {}".format(self.observation_space.high))
        print("\tObservation space low : {}".format(self.observation_space.low))

        print("\tOriginal action space object type:", type(self.env.action_space))

        # ACTION SPACE
        # Need conditions for different action structure of different classes
        if type(self.env.action_space) is gym.spaces.discrete.Discrete:
            assert env_details[1] == 'discrete'
            assert env_details[2] == 0
            self.action_space = np.arange(self.env.action_space.n)
            self.action_space_n = self.env.action_space.n
        elif type(self.env.action_space) in \
                [gym.spaces.box.Box, quanser_robots.common.LabeledBox]:
            assert env_details[1] == 'continuous'
            self.action_space = np.linspace(self.env.action_space.low,
                                            self.env.action_space.high,
                                            env_details[2]) # num of actions
            self.action_space_n = env_details[2]
        else:
            raise ValueError("Env Action Space should be of type Discrete "
                             "or Box, but is of Type {}."
                             .format(type(self.env.action_space)))

        self.env.action_space = self.action_space
        self.action_space_high = np.max(self.action_space)
        self.action_space_low = np.min(self.action_space)

        print("\tAction space high: {}".format(self.action_space_high))
        print("\taction space low : {}".format(self.action_space_low))
        print("\tAction space: {}".format(self.action_space.tolist()))


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
