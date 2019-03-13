import numpy as np
import random
import gym
import quanser_robots
import warnings
import sys


class NAC:
    """

    """

    def __init__(self, env, actor, critic):
        self.env = env
        self.actor = actor
        self.critic = critic

    def run_batch(self, sess):
        """


        :param sess:
        :return:
        """

        # Reset the environment and get start state
        observation = self.env.reset()

        # Variables saving data for the current trajectory
        traj_reward = 0.0
        traj_transitions = []

        # Variables saving data for the complete batch
        batch_traj_rewards = []
        batch_states = []
        batch_actions = []
        batch_advantages = []
        batch_discounted_returns = []

        for t in range(self.env.time_steps):
            # Some environments need preprocessing
            observation = self.preprocess_obs(observation)

            # ------------------- PREDICT ACTION ---------------------------- #
            if self.env.continuous:
                # Not implemented yet
                obs_vector = np.expand_dims(observation, axis=0)
                (act_state_input, _, _, act_probabilities, _) \
                    = self.actor.get_net_variables()
                action = sess.run(
                    act_probabilities,
                    feed_dict={act_state_input: obs_vector})
                batch_actions.append(action)

            else:
                # Get the action with the highest probability w.r.t our actor
                action, action_i = self.actor.get_action(sess, observation)

                # Make one-hot action array
                action_array = np.zeros(len(self.env.action_space))
                action_array[action_i] = 1
                batch_actions.append(action_array)

            # --------------- TAKE A STEP IN THE ENV ------------------------ #

            old_observation = observation
            observation, reward, done, _ = self.env.step(action)

            # Record state/transition
            batch_states.append(old_observation)
            traj_transitions.append((old_observation, action, reward))
            traj_reward += reward

            # -------------------- END OF TRAJECTORY ------------------------ #

            # If env = done or we collected our desired number of steps
            if done or t == self.env.time_steps - 1:

                discounted_return = 0.0
                traj_advantages = []
                traj_discounted_returns = []

                # Calculate in reverse order, because it's faster
                for trans_i in reversed(range(len(traj_transitions))):
                    obs, action, reward = traj_transitions[trans_i]

                    # ------- Discounted monte-carlo return (G_t) ----------- #

                    discounted_return = discounted_return * \
                                        self.env.mc_discount_factor + reward

                    # Save disc reward to update critic params in its direction
                    traj_discounted_returns.insert(0, discounted_return)

                    # ------------------- ADVANTAGE ------------------------- #

                    # Get the value V from our Critic
                    critic_value = self.critic.estimate(sess, obs)

                    # Save advantages to update actor params in its direction
                    traj_advantages.insert(0, discounted_return - critic_value)

                # ----------------- SAVE VARIABLES -------------------------- #

                batch_discounted_returns.extend(traj_discounted_returns)
                batch_advantages.extend(traj_advantages)
                batch_traj_rewards.append(traj_reward)

                # ----------------- RESET VARIABLES ------------------------- #
                traj_reward = 0.0
                traj_transitions = []

                if done:
                    # Reset environment, if we still have steps left in batch
                    observation = self.env.reset()
                else:
                    # If we have no steps left, close environment
                    self.env.close()

        # ----------------- UPDATE NETWORKS -------------------------------- #

        self.critic.update(sess, batch_states, batch_discounted_returns)
        self.actor.update(sess, batch_states, batch_actions, batch_advantages)

        return batch_traj_rewards

    def preprocess_obs(self, observation):
        """
        We need to preprocess our observations for two reasons
        1. We do not want to have zeros, because it is not feasible with our
        way of calculating the fisher inverse (dividing by zero).
        2. Some environments have some constraints or function better if the
        state (specific w.r.t. the environment) is clipped.

        :param observation: the observation we want to preprocess
        :return: the preprocessed observation
        """

        observation = [0.00001 if np.abs(x) < 0.00001 else x for x in observation]

        if self.env.name == 'Qube-v0':
            for rr in range(4):
                ob = observation[rr]
                if ob > 0.999:
                    observation[rr] = 0.999
                elif ob < -0.999:
                    observation[rr] = -0.999

        return observation

