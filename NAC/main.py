import tensorflow as tf
import numpy as np
import sys
import time
import gym
import random

from my_environment import MyEnvironment
from critic import Critic
from actor import Actor
from nac import NAC
import evaluation

# ----------------------------- GOALS ------------------------------------- #

# Environments which have to be solved:
# DoubleCartPole: "DoublePendulum-v0"
# FurutaPend: "Qube-v0"
# BallBalancer: "BallBalancerSim-v0"

# ---------------------- VARIABLES & CONSTANTS ------------------------------ #
# Do we want to render the environment after evaluation?
RENDER = True

# Select how we treat actions
# IMPORTANT: Only False works yet
CONTINUOUS = False
HIDDEN_LAYER_SIZE = 10

# Select complexity of policy network
# IMPORTANT: Only False works yet
COMPLEX_POLICY_NET = False

# Load weights from file and use them
LOAD_WEIGHTS = False

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
"""

env_dict = {1: ['CartPole-v0',          'discrete',     [0],    200, 300, 0.97, 0.001, 0.1],


            2: ['DoublePendulum-v0',    'continuous',   [3],    200, 300, 0.97, 0.001, 0.1],
                # Does not diverge with batch size of 2000

            3: ['Qube-v0',              'continuous',   [3],    200, 300, 0.97, 0.001, 0.1],
            4: ['BallBalancerSim-v0',   'continuous',   [5, 5], 4000, 300, 1, 0.001, 0.1],
            5: ['Levitation-v1',        'continuous',   [3],    200, 300, 0.97, 0.001, 0.1],
            6: ['Pendulum-v0',          'continuous',   [3],    200, 300, 0.97, 0.001, 0.1],
            11: ['CartpoleStabRR-v0',   'discrete',     [0],    500, 300, 0.97, 0.001, 0.1]}

assert ENVIRONMENT in env_dict.keys()
env_details = env_dict[ENVIRONMENT]

# --------------------------------------------------------------------------- #



if LOAD_WEIGHTS:

    sess = tf.InteractiveSession()
    saver = tf.train.import_meta_graph('model/nac_model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('model/'))
    graph = tf.get_default_graph()

    pl_probabilities = graph.get_tensor_by_name("actor/probabilities:0")
    pl_state_input = graph.get_tensor_by_name("actor/state_input:0")


    # ---------------------- GENERATE ENVIRONMENT ------------------------------- #
    print("Generating {} environment:".format(env_details[0]))
    env = gym.make(env_details[0])

    # # ----------------------- GENERATE NETWORKS --------------------------------- #

    time_steps = 10000
    for e in range(10):

        print("Episode {} ... ".format(e), end='')
        sys.stdout.flush()

        done = False
        observation = env.reset()

        for t in range(time_steps):

            # Render environment
            env.render()
            time.sleep(0.01)

            # Break loop, if episode has finished
            if done:
                print("Episode ended after {} time steps!".format(t))
                break

            # Get probabilites of actions to take
            obs_vector = np.expand_dims(observation, axis=0)
            probs = sess.run(
                pl_probabilities,
                feed_dict={pl_state_input: obs_vector})

            # Stochastically generate an action using the policy output probs
            probs_sum = 0
            action_i = None  # Action index
            rnd = random.uniform(0, 1)
            for k in range(len(env.action_space)):
                probs_sum += probs[0][k]
                if rnd < probs_sum:
                    action_i = k
                    break
                elif k == (len(env.action_space) - 1):
                    action_i = k
                    break

            action = env.action_space[action_i]

            observation, _, done, _ = env.step(action)


    sess.close()


else:

    # Initialize the session
    sess = tf.InteractiveSession()

    # ---------------------- GENERATE ENVIRONMENT ------------------------------- #
    print("Generating {} environment:".format(env_details[0]))
    env = MyEnvironment(env_details, CONTINUOUS,
                        COMPLEX_POLICY_NET, HIDDEN_LAYER_SIZE)

    # ----------------------- GENERATE NETWORKS --------------------------------- #

    print("Generating Neural Networks ... ", end="")
    start_time = time.time()
    sys.stdout.flush()
    actor = Actor(env)
    critic = Critic(env)
    nac = NAC(env, actor, critic)
    env.network_generation_time = int(time.time() - start_time)
    print("Done! (Time: " + str(env.network_generation_time) + " seconds)")

    sess.run(tf.global_variables_initializer())

    # ----------------------- TRAINING NETWORKS --------------------------------- #

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
    print('Average time: %.3f' % (np.sum(times) / env.num_of_updates))

    # ------------------- EVALUATE | SAVE | RENDER ------------------------------ #

    evaluation.evaluate(env, sess, actor)

    saver = tf.train.Saver()
    saver.save(sess, '{}/model/nac_model'.format(env.save_folder))

    if RENDER:
        evaluation.render(env, sess, actor)

    sess.close()
