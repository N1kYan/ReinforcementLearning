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
import quanser_robots

from NAC_by_studywolf.my_environment import MyEnvironment
from NAC_by_studywolf.critic_network import value_gradient
from NAC_by_studywolf.actor_network import Actor
from NAC_by_studywolf.natural_policy_gradient import run_batch
from NAC_by_studywolf import Evaluation

# ----------------------------- GOALS ------------------------------------- #

# Environments which have to be solved:
# DoubleCartPole: "DoublePendulum-v0"
# FurutaPend: "Qube-v0"
# BallBalancer: "BallBalancerSim-v0"
# Levitation: "Levitation-v1"

# ----------------------------- TODOS --------------------------------------- #
# TODO: Print Results to a file including parameters
# TODO: Continuous actions
# TODO: Improvements: https://github.com/rgilman33/simple-A2C/blob/master/3_A2C-nstep-TUTORIAL.ipynb
# TODO: Some paper said, not taking batches, where we only had 1 episode because
#   it does not yield any improvements to update the weights of perffect episodes.

# ---------------------- VARIABLES & CONSTANTS ------------------------------ #
# Select Rendering
RENDER = True

# Select how we treat actions
# IMPORTANT: Only False works yet
CONTINUOUS = False

# Select complexity of policy network
# IMPORTANT: Only False works yet
COMPLEX_POLICY_NET = False

# Select Environment
ENVIRONMENT = 1

"""
    0: Name of the Gym/Quanser environment.
    1: If the environment is descrete or continuous.
    2: Chose the discretization of continuous environments (discrete = 0).
       Only important, if CONTINUOUS = False.
    3: Batch size. How much steps should the agent perform before updating 
       parameters. If the trajectory ends before that (done == True), a new 
       trajectory is started.
    4: How many updates (of parameters) do we want.
    5: Discount factor for expected monte carlo return.
"""

env_dict = {1: ['CartPole-v0',          'discrete',     0, 200, 300, 0.97],

            2: ['DoublePendulum-v0',    'continuous',   9, 2000, 300, 0.97],
                # Does not diverge with batch size of 2000

            3: ['Qube-v0',              'continuous',   3, 200, 300, 0.97],
            4: ['BallBalancerSim-v0',   'continuous',   3, 200, 300, 0.97],
            5: ['Levitation-v1',        'continuous',   5, 200, 300, 0.97],
            6: ['Pendulum-v0',          'continuous',   3, 200, 300, 0.97],
            7: ['CartpoleStabRR-v0',    'continuous',   3, 200, 300, 0.97]}

assert ENVIRONMENT in env_dict.keys()
env_details = env_dict[ENVIRONMENT]


# ---------------------- GENERATE ENVIRONMENT ------------------------------- #
print("Generating {} environment:".format(env_details[0]))
env = MyEnvironment(env_details=env_details)




# ----------------------- TRAINING NETWORKS --------------------------------- #
# We run the same algorithm 10 times and save the results
for run in range(1):

    # Initialize the session
    sess = tf.InteractiveSession()

    # ----------------------- GENERATE NETWORKS ----------------------------- #

    print("Generating Neural Networks ... ", end="")
    sys.stdout.flush()
    actor = Actor(env, CONTINUOUS, COMPLEX_POLICY_NET)
    value_grad = value_gradient(env)
    print("Done!")

    sess.run(tf.global_variables_initializer())

    max_rewards = []
    total_episodes = []
    times = []

    num_of_updates = env_details[4]

    for u in range(num_of_updates):
        start_time = time.time()

        # Act in the env and update weights after collecting data
        reward, n_episodes = \
            run_batch(env, actor, value_grad, sess, u, CONTINUOUS) # TODO

        max_rewards.append(np.max(reward))
        total_episodes.append(n_episodes)
        times.append(time.time() - start_time)
    print('Average time: %.3f' % (np.sum(times) / num_of_updates))

    Evaluation.evaluate(env, sess, actor)

    if RENDER:
        Evaluation.render(env, sess, actor)

    sess.close()
