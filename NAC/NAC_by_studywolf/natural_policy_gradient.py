import numpy as np
import random
import gym
import quanser_robots
import warnings


def run_episode(env, policy_grad, value_grad, sess, num_traj, printing=False):

    # Unpack the policy network (generates control policy)
    (pl_state, pl_actions, pl_advantages,
        pl_probabilities, pl_train_vars) = policy_grad

    # Unpack the value network (estimates expected reward)
    (vfa_state_input, vfa_true_vf_input,
        vfa_nn_output, vfa_optimizer, vfa_loss) = value_grad

    # set up the environment
    observation = env.reset()

    episode_reward = 0
    total_rewards = []
    states = []
    actions = []
    advantages = []
    transitions = []
    update_vals = []

    n_episodes = 0
    n_timesteps = env.time_steps

    # calculate policy
    for t in range(n_timesteps):

        # I think sometimes we have a zero in the observations
        # and we somehow divide while calculating the probs
        observation = \
            [0.00001 if np.abs(x) < 0.00001 else x for x in observation]

        if env.name == 'Qube-v0':
            for rr in range(4):
                ob = observation[rr]
                if ob > 0.999:
                    observation[rr] = 0.999
                elif ob < -0.999:
                    observation[rr] = -0.999

        # Expand state by one dimension
        # Before = [ ... , ... , ... ], After = [[ ... , ... , ... ]]
        obs_vector = np.expand_dims(observation, axis=0)

        print("({}) OBS:{}".format(t, obs_vector), end='') if printing else ...

        # ------------------- PREDICT ACTION -------------------------------- #
        probs = sess.run(
            pl_probabilities,
            feed_dict={pl_state: obs_vector})

        print(", PROBS:", probs, end='') if printing else ...

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
        print(", ACTION ARRAY: ", action_array, end='') if printing else ...

        old_observation = observation

        # Get the action (not only the index)
        # and take the action in the environment
        # Try/Except: Some env need action in an array
        action = env.action_space[action_i]  # TODO: Actions sind nicht immer 1D

        print(", ACTION-1:", action, end='') if printing else ...

        try:
            observation, reward, done, info = env.step(action)
        except AssertionError:
            action = np.array([action])
            observation, reward, done, info = env.step(action)

        print(", ACTION-2: ", action) if printing else ...

        transitions.append((old_observation, action, reward))
        episode_reward += reward

        # ---------------- End of trajectory -------------------------------- #

        # If the pole falls or we collected our number of steps
        if done or t == n_timesteps - 1:
            for o, trans in enumerate(transitions):
                obs, action, reward = trans

                # Calculate discounted monte-carlo return
                future_reward = 0
                future_transitions = len(transitions) - o
                decrease = 1
                for p in range(future_transitions):
                    future_reward += transitions[p + o][2] * decrease
                    decrease = decrease * env.discount_factor

                obs_vector = np.expand_dims(obs, axis=0)
                # compare the calculated expected reward to the average
                # expected reward, as estimated by the value network
                current_val = sess.run(
                    vfa_nn_output,
                    feed_dict={vfa_state_input: obs_vector}
                )[0][0]

                # advantage: how much better was this action than normal
                advantages.append(future_reward - current_val)

                # update the value function towards new return
                update_vals.append(future_reward)

            n_episodes += 1
            # reset variables for next episode in batch
            total_rewards.append(episode_reward)
            episode_reward = 0.0
            transitions = []

            if done:
                # if the pole fell, reset environment
                observation = env.reset()
            else:
                # if out of time, close environment
                env.close()

    print("\n\n\n") if printing else ...
    print('Update {} with {} trajectories with rewards of: {}'
          .format(num_traj, len(total_rewards), total_rewards))

    # update value function
    update_vals_vector = np.expand_dims(update_vals, axis=1)
    sess.run(vfa_optimizer,
             feed_dict={vfa_state_input: states,
                        vfa_true_vf_input: update_vals_vector})

    # update control policy
    sess.run(pl_train_vars,
             feed_dict={pl_state: states,
                        pl_advantages: advantages,
                        pl_actions: actions})

    return total_rewards, n_episodes
