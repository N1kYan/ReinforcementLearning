""" Copyright (C) 2018 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.


Learning cart-pole with a policy network and value network.
The policy network learns the actions to take, and the value network learns
the expected reward from a given state, for use in calculating the advantage
function.

The value network is updated with a 2 norm loss * .5 without the sqrt on the
difference from the calculated expected cost. The value network is used to
calculated the advantage function at each point in time.

The policy network is updated by calculating the natural policy gradient and
advantage function, with a learning rate calculated to normalize the KL
divergence of the policy network output. This works to prevent any parameter
updates from drastically changing behaviour of the system.

Adapted from KVFrans:
github.com/kvfrans/openai-cartpole/blob/master/cartpole-policygradient.py
"""

import tensorflow as tf
import numpy as np
import sys
import time
import gym
# import quanser_robots

from NAC_by_studywolf.my_environment import MyEnvironment
from NAC_by_studywolf.critic_network import value_gradient
from NAC_by_studywolf.actor_network import policy_gradient
from NAC_by_studywolf.natural_policy_gradient import run_episode
from NAC_by_studywolf import Evaluation

# Environments which have to be solved:
# DoubleCartPole: "DoublePendulum-v0"
# FurutaPend: "Qube-v0"
# BallBalancer: "BallBalancerSim-v0"

# ----------------------------- RESULTS ------------------------------------- #
# DoublePend, 3 actions, 300 Traj, 300 timesteps it holds the sticks far longer
# (avg. 450) than if we have 200 timesteps with same config (avg. 200)

# ---------------------- VARIABLES & CONSTANTS ------------------------------ #
PRINTING = False

# How much steps should the agent perform before updating parameters
TIME_STEPS = 200

# Number of trajectories, where each is of length TIME_STEPS
# Trajectories consist of one or several environment episodes
N_TRAJECTORIES = 300

# Select Environment
ENVIRONMENT = 1

# Set the discretization of continuous environments here
env_dict = {1: ['CartPole-v0',          'discrete'],
            2: ['DoublePendulum-v0',    'continuous',   3],
            3: ['Qube-v0',              'continuous',   3],
            4: ['BallBalancerSim-v0',   'discrete'],
            5: ['Levitation-v1',   'discrete'],
            6: ['Pendulum-v0',          'continuous',   3],
            7: ['CartpoleSwingShort-v0','continuous',   3],
            8: ['CartpoleStabRR-v0',    'continuous',   3]}
assert ENVIRONMENT in env_dict.keys()

# ---------------------- GENERATE ENVIRONMENT ------------------------------- #
print("Generating {} environment:".format(ENVIRONMENT), end="")
num_actions = env_dict[ENVIRONMENT]
if NUM_ACTIONS is not None and num_actions is not None:
    num_actions = NUM_ACTIONS
env = MyEnvironment(ENVIRONMENT, num_actions, TIME_STEPS)

# ----------------------- GENERATE NETWORKS --------------------------------- #

print("Generating Neural Networks ... ", end="")
sys.stdout.flush()
policy_grad = policy_gradient(env)
value_grad = value_gradient(env)
print("Done!")

# run the training from scratch 10 times, record results
for ii in range(1):
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    max_rewards = []
    total_episodes = []
    times = []
    for i in range(N_TRAJECTORIES):
        start_time = time.time()
        reward, n_episodes = \
            run_episode(env, policy_grad, value_grad, sess, i,
                        printing=PRINTING)
        max_rewards.append(np.max(reward))
        total_episodes.append(n_episodes)
        times.append(time.time() - start_time)
    print('Average time: %.3f' % (np.sum(times) / N_TRAJECTORIES))

    np.savez_compressed('data/natural_policy_gradient_%i' % ii,
                        max_rewards=max_rewards, total_episodes=total_episodes)

    # Render the result
    Evaluation.evaluate(env, policy_grad, 50, True, 0.1, sess)

    sess.close()
