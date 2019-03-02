import gym
import quanser_robots
import numpy as np
import tensorflow as tf
import random

from REINFORCE_ALEX.Policy_Network import policy_gradient


class REINFORCEAgent:

    def __init__(self, env, learning_rate, discounting, load_weights):
        self.env = env
        self.load_weights_flag = load_weights
        self.gamma = discounting
        self.learning_rate = learning_rate

        # Remember episode
        self.trajectory_states, self.trajectory_actions, \
        self.trajectory_probs, self.trajectory_rewards \
            = [], [], [], []

        # Get the network structure (gradient based)
        self.policy_grad = policy_gradient(env, learning_rate)

    def remember(self, state, action, prob, reward):
        """
        Saving a trajectory.
        :param state: current state s of the environment
        :param action: current performed action a
        :param reward: current observed reward r
        :return: none
        """
        self.trajectory_states.append(state)
        self.trajectory_actions.append(action)
        self.trajectory_probs.append(prob)
        self.trajectory_rewards.append(reward)

    def discounted_rewards(self, rewards):
        """
        Calculate discounted rewards from trajectory.
        :param rewards: Remembered rewards from trajectory (per time-step)
        :return:
        """
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            # The following condition makes no sense to me (Alex)
            # if rewards[t] != 0:
            #     running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def run_episode(self, sess, num_traj):
        # unpack the policy network (generates control policy)
        (pl_state, pl_actions, pl_advantages,
         pl_probabilities, pl_train_vars) = self.policy_grad

        # set up the environment
        observation = env.reset()

        episode_reward = 0
        states = []
        actions = []
        advantages = []
        rewards = []

        done = False

        # calculate policy
        while not done:

            # Expand state by one dimension
            # Before = [ ... , ... , ... ], After = [[ ... , ... , ... ]]
            obs_vector = np.expand_dims(observation, axis=0)

            # Probabilities
            probs = sess.run(
                pl_probabilities,
                feed_dict={pl_state: obs_vector})

            # Check which action to take
            # stochastically generate action using the policy output
            probs_sum = 0
            action_i = None
            rnd = random.uniform(0, 1)
            for k in range(len(env.action_space)):
                probs_sum += probs[0][k]
                if rnd < probs_sum:
                    action_i = k
                    break
                elif k == (len(env.action_space) - 1):
                    action_i = k
                    break

            # record the transition
            states.append(observation)
            # Make one-hot action array
            action_array = np.zeros(len(env.action_space))
            action_array[action_i] = 1
            actions.append(action_array)

            old_observation = observation

            # Get the action (not only the index)
            # and take the action in the environment
            # Try/Except: Some env need action in an array
            action = env.action_space[action_i]
            try:
                observation, reward, done, info = env.step(action)
            except AssertionError:
                action = np.array([action])
                observation, reward, done, info = env.step(action)

            rewards.extend(reward)
            episode_reward += reward

            # if the pole falls or time is up
            if done:

                # reset variables for next episode in batch
                episode_reward = 0.0
                transitions = []

                observation = env.reset()

        print('Trajectory: {}, Reward: {}'.format(num_traj, episode_reward))

        # update control policy
        sess.run(pl_train_vars,
                 feed_dict={pl_state: states,
                            pl_advantages: advantages,
                            pl_actions: actions})

        return episode_reward

