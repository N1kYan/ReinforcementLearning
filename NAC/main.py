import tensorflow as tf
import numpy as np
import sys
import time
import gym
import random
import os

from my_environment import MyEnvironment
from critic import Critic
from actor import Actor
from nac import NAC
import evaluation

# ---------------------- VARIABLES & CONSTANTS ------------------------------ #
# Do we want to render the environment after evaluation?
RENDER = True

# Load weights from file and use them
LOAD_WEIGHTS = False

TRAIN = True
EVALUATION = True

# -------------------------- ENVIRONMENT ------------------------------------ #

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
    6: Learning rate for actor model. sqrt(learning_rate/ Grad_j^T * F^-1).
    7: Learning rate for Adam optimizer in the critic model.
    8: The hidden layer size of the critic network.
"""

env_dict = {
    1:  ['CartPole-v0', 'discrete', [0],
         500, 300, 0.97, 0.001, 0.1, 10],

    2:  ['DoublePendulum-v0', 'continuous', [3],
         200, 300, 0.97, 0.001, 0.1, 10],

    3:  ['Qube-v0', 'continuous', [3],
         200, 300, 0.97, 0.001, 0.1, 10],

    4:  ['BallBalancerSim-v0', 'continuous', [9, 9],
         2000, 300, 1,   0.001, 0.1, 10],

    5:  ['Levitation-v1', 'continuous', [3],
         200, 300, 0.97, 0.001, 0.1, 10],

    6:  ['Pendulum-v0', 'continuous', [3],
         200, 300, 0.97, 0.001, 0.1, 10],

    11: ['CartpoleStabRR-v0', 'discrete', [0],
         500, 300, 0.97, 0.001, 0.1, 10]
}

assert ENVIRONMENT in env_dict.keys()
env_details = env_dict[ENVIRONMENT]

# --------------------------------------------------------------------------- #
# Initialize the session
sess = tf.InteractiveSession()

# ---------------------- GENERATE ENVIRONMENT ------------------------------- #
print("Generating {} environment:".format(env_details[0]))
env = MyEnvironment(env_details)

# ----------------------- GENERATE NETWORKS --------------------------------- #
hour, min, sec = time.strftime("%H,%M,%S").split(',')
print("Generating Neural Networks (Time: {}:{}:{}) ... "
      .format(hour, min, sec), end="")
sys.stdout.flush()
start_time = time.time()

if LOAD_WEIGHTS:
    saver = tf.train.import_meta_graph('model/nac_model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('model/'))
    graph = tf.get_default_graph()

    a_state_input = graph.get_tensor_by_name("actor/state_input:0")
    a_actions_input = graph.get_tensor_by_name("actor/actions_input:0")
    a_advantages_input = \
        graph.get_tensor_by_name("actor/advantages_input:0")
    a_probabilities = graph.get_tensor_by_name("actor/probabilities:0")
    a_weights = graph.get_tensor_by_name("actor/weights:0")

    c_state_input = graph.get_tensor_by_name("critic/state_input:0")
    c_true_vf_input = graph.get_tensor_by_name("critic/true_vf_input:0")
    c_output = graph.get_tensor_by_name("critic/output:0")
    c_optimizer = tf.get_collection("optimizer")
    c_loss = graph.get_tensor_by_name("critic/loss:0")

else:
    a_state_input, a_actions_input, a_advantages_input, \
        a_probabilities, a_weights = Actor.create_policy_net(env)

    c_state_input, c_true_vf_input, c_output, c_optimizer, c_loss = \
        Critic.create_value_net(env)

actor = Actor(env, a_state_input, a_actions_input, a_advantages_input,
              a_probabilities, a_weights)
critic = Critic(env, c_state_input, c_true_vf_input, c_output,
                c_optimizer, c_loss)
nac = NAC(env, actor, critic)

if not LOAD_WEIGHTS:
    sess.run(tf.global_variables_initializer())

env.network_generation_time = int(time.time() - start_time)
print("Done! (Time: " + str(env.network_generation_time) + " seconds)")


# ----------------------- TRAINING NETWORKS --------------------------------- #
if TRAIN:
    max_rewards = []
    total_episodes = []
    times = []

    for u in range(env.num_of_updates):
        start_time = time.time()

        # Act in the env and update weights after collecting data
        batch_traj_rewards = nac.run_batch(sess)

        print('Update {} with {} trajectories with rewards of: {}'
              .format(u, len(batch_traj_rewards), batch_traj_rewards))

        max_rewards.append(np.max(batch_traj_rewards))
        total_episodes.append(len(batch_traj_rewards))
        times.append(time.time() - start_time)

    # Save model to file
    try:
        os.makedirs(env.save_folder)
    except FileExistsError:
        pass
    saver = tf.train.Saver()
    saver.save(sess, '{}/model/nac_model'.format(env.save_folder))

    print('Average time: %.3f' % (np.sum(times) / env.num_of_updates))

# ------------------- EVALUATE | RENDER ------------------------------ #

if EVALUATION:
    evaluation.evaluate(env, sess, actor)

if RENDER:
    evaluation.render(env, sess, actor)

sess.close()
