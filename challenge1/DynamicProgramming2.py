import numpy as np
import TrueModel
import gym
import Utils
import sys
import Evaluation
import pickle
import os.path
import errno


"""
    Alex wants to implement value iteration again  which is  why he created  
    this class.
"""


# PARAMETER
STATE_SPACE_SIZE = (16 + 1, 16 + 1)
ACTION_SPACE_SIZE = (16 + 1)
THETA = 0.1
GAMMA = 0.8
STOCHASTIC = False # Do we want stochastic or deterministic transitions
SIGMA = 1 # For Gauss function (stochastic transitions)

# In the stochastic case we take the rewards of the discrete state we could
# land in. Because we do not get this from our environment (we just get the
# reward of the continuous state  we land in), we have to estimate it before
# we execute stochastic value iteration.
# If CALC_DISC_REWARDS is True, we calculate these rewards before we start the
# value iteration (TIME CONSUMING). If not, we load it from a pickle file, so
# it has to be executed at least once before we start value iteration.
CALC_DISC_REWARDS = False

# Variables
env = gym.make('Pendulum-v2') # Create gym/quanser environment
action_space = (np.linspace(env.action_space.low, env.action_space.high,
                            ACTION_SPACE_SIZE))
state_space = (np.linspace(-np.pi, np.pi, STATE_SPACE_SIZE[0]),
               np.linspace(-8, 8, STATE_SPACE_SIZE[1]))

# Error list for discrete rewards that weren't found
# Lists in a list. Inner list: 1. and 2. entry is the state, 3. entry is
# the number of times we wanted to access this reward
# [[radians, velocity, counter], ... ].
reward_is_none = []



def discretize_index(state):

    radian_diff =  [np.abs(x - state[0]) for x in state_space[0]]
    vel_diff = [np.abs(x - state[1]) for x in state_space[1]]

    index = np.array([np.argmin(radian_diff), np.argmin(vel_diff)])

    return index


def gaussian(x, mean, std):
    return (1/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*np.square((x-mean)/std))

# input state: continous
# e.g. std=1
def get_probs(space, state, std):
    probs = {}
    top_border = state + 3 * std
    if top_border[0] > np.pi:
        top_border -= 2 * np.pi

    bot_border = state - 3 * std
    if bot_border[0] < -np.pi:
        bot_border += 2 * np.pi

    #TODO: Scale to 100

    max_index = discretize_index(top_border)[0]
    min_index = discretize_index(bot_border)[0]

    # if we go from 3.1 to -3.1 its only a small step
    # so we have to go from top index to bot index
    if top_border[0] < bot_border[0]:
        intervall_1 = np.arange(min_index, len(space[0]))
        intervall_2 = np.arange(0, max_index)
        i_intervall = np.concatenate((intervall_1, intervall_2))
    else:
        i_intervall = np.arange(min_index,max_index+1)

    probs_list = []

    for i in i_intervall:
        s = space[0][i]
        if s < 0:
            s_hat = s + 2*np.pi
            if (s_hat - state[0]) < (s - state[0]):
                s = s_hat
        else:
            s_hat = s - 2*np.pi
            if (s_hat - state[0]) < (s - state[0]):
                s = s_hat

        gaus = gaussian(s, state[0], std)
        # print(gaus)

        probs_list.append(gaus)

    probs_list = [x * (1/np.sum(probs_list)) for x in probs_list]
    return i_intervall, probs_list


def learn_discrete_rewards():

    print("Learning discrete rewards ... ")
    sys.stdout.flush()

    r_dash = np.full(STATE_SPACE_SIZE, None)
    env.reset()

    num_of_estimations = 0
    rounds_since_last_finding = 0

    done = False

    while(True):
        if done:
            env.reset()
        a = env.action_space.sample()
        next_state, reward, done, info = env.step(a)
        for j, s0 in enumerate(state_space[0]):
            for k, s1 in enumerate(state_space[1]):
                # If we already have a saved value (r is not default value), we
                # don't need to compare the current next_state with the state
                # underlying r.

                if r_dash[j][k] is None:
                    if (np.abs(s0 - next_state[0]) < 0.1) and \
                            (np.abs(s1  - next_state[1]) < 0.1):
                        r_dash[j][k] = reward

                        num_of_estimations += 1
                        rounds_since_last_finding = 0
                        print("Number of estimations: ", num_of_estimations)

        if num_of_estimations == r_dash.size:
            break

        rounds_since_last_finding += 1
        if rounds_since_last_finding == 1000000:
            break


    with open('pickle/discrete_rewards.pkl', 'wb') as handle:
        pickle.dump(r_dash, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("... Done!")
    print("Estimated {} states.".format(num_of_estimations))
    print(r_dash)

    return True


# TODO: Stochastic reward is only 1 dimensional, also we have a 2-dim state
# For now we just care about the first dimension (degree in radians) and
# copy the second dimension (velocity) as it is in the state we are coming from
def stochastic_reward(state_space, s_dash, V, true_model_reward, discrete_rewards, sigma):
    exp_reward = 0
    index = discretize_index(s_dash)
    s_indices, probs = get_probs(state_space, s_dash, sigma)


    #iterate over all possible states
    for i, s_index in enumerate(s_indices):
        immediate_reward = discrete_rewards[s_index][index[1]]
        # If in our simulation of the environment no action landed in state
        # state[s_index][index[1], we have no reward for the landing state
        # instead we will take the true_model_reward, also we know it may
        # differ from the true reward of the state we discretize our continuous
        # state to.
        # For inspection purposes, print a warning!
        if immediate_reward is None:
            velocity = state_space[1][index[1]]
            # We just have for velocity 6-8 no rewards.
            # So we try to choose a velocity a little more to 0 which has a
            # reward. The velocity is anyway just copied from the starting
            # state.
            for num in range(1,4):
                if velocity > 0:
                    immediate_reward = discrete_rewards[s_index][index[1]-num]
                else:
                    immediate_reward = discrete_rewards[s_index][index[1]+num]

                if immediate_reward is not None:
                    break
            if immediate_reward is None:
                immediate_reward = true_model_reward

            # For debugging/printing put in global list
            not_yet_in_list = True
            for l, rin in enumerate(reward_is_none):
                if rin[0] == state_space[0][s_index] \
                    and rin[1] == state_space[1][index[1]]:
                    reward_is_none[l] = [rin[0], rin[1], rin[2]+1]
                    not_yet_in_list = False
                    break
            if not_yet_in_list:
                reward_is_none.append(
                    [state_space[0][s_index], state_space[1][index[1]], 1])




        V_dash = V[s_index, index[1]]
        # print(V_dash)
        R = immediate_reward + (GAMMA * V_dash)
        # print(R)
        # print(probs[i])
        exp_reward += probs[i] * R

    return exp_reward


# def stochastic_reward_3states(state_space, main_probability, index,
#                                   immediate_reward, V):
#
#     exp_reward = 0
#     V_dash = V[index[0], index[1]]
#
#     # -200 is discounted and therefore gets better
#     # 0.8 * -200 = -160
#     R = immediate_reward + (GAMMA * V_dash)
#     exp_reward += main_probability * R
#
#     index[0] = (index[0]+1) % len(state_space[0])
#     V_dash = V[index[0], index[1]]
#     R = immediate_reward + (GAMMA * V_dash)
#     exp_reward += (1-main_probability)/2 * R
#
#     index[0] = (index[0]-2) % len(state_space[0])
#     V_dash = V[index[0], index[1]]
#     R = immediate_reward + (GAMMA * V_dash)
#     exp_reward += (1-main_probability)/2 * R
#
#     return exp_reward



def value_iteration(state_space, action_space, stochastic=True, sigma=1):

    discrete_rewards = None
    if stochastic:
        filename = 'pickle/discrete_rewards.pkl'
        if os.path.isfile(filename):
            with open(filename, 'rb') as handle:
                discrete_rewards = pickle.load(handle)
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), filename)

    V = np.full(shape=STATE_SPACE_SIZE, fill_value=0)

    loop_num = 0

    while(True):

        loop_num += 1
        print("Getting VF, Loop {} ... ".format(loop_num), end='')
        sys.stdout.flush()
        delta = 0.0

        for i_s0, s0 in enumerate(state_space[0]):
            for i_s1, s1 in enumerate(state_space[1]):
                v = V[i_s0][i_s1]

                R_list = []

                for a in action_space:
                    s_dash = TrueModel.transition([s0, s1, a])
                    # # print("next state: ", s_dash)
                    true_model_reward = TrueModel.reward([s0, s1, a])

                    if stochastic:
                        R = stochastic_reward(state_space, s_dash, V, true_model_reward, discrete_rewards, sigma)
                    else:
                        index = discretize_index(s_dash)
                        V_dash = V[index[0], index[1]]
                        R = true_model_reward + (GAMMA * V_dash)

                    R_list.append(R)

                V[i_s0][i_s1] = np.max(R_list)

                # print("-- ", i_s0, ", ", i_s1, " --")
                # print(V[i_s0][i_s1])
                # print(v)

                v_V_diff = np.abs(v - V[i_s0][i_s1])

                delta = np.max([delta, v_V_diff])

        print("Done! (Delta =  {})".format(delta))

        if delta < THETA:
            break

    print("--------------- Value Function --------------------")
    print(V)
    print("---------------------------------------------------")

    # POLICY
    print("Getting policy ... ", end='')
    sys.stdout.flush()
    pi = np.full(shape=STATE_SPACE_SIZE, fill_value=-3)
    for i_s0, s0 in enumerate(state_space[0]):
        for i_s1, s1 in enumerate(state_space[1]):

            R_list = []

            for a in action_space:
                s_dash = TrueModel.transition([s0, s1, a])
                # # print("next state: ", s_dash)
                true_model_reward = TrueModel.reward([s0, s1, a])

                if stochastic:
                    R = stochastic_reward(state_space, s_dash, V,
                                          true_model_reward, discrete_rewards,
                                          sigma)
                else:
                    index = discretize_index(s_dash)
                    V_dash = V[index[0], index[1]]
                    R = true_model_reward + (GAMMA * V_dash)

                R_list.append(R)

            # print("-- ", i_s0, ", ", i_s1, ", ", a, " --")
            # print("s0: ", s0, ", s1: ", s1, ", a: ", a)
            # print("s_dash: ", s_dash, " index: ", index, " V_dash: ", V_dash)

            # print(i_s0, i_s1, R_list)

            pi[i_s0][i_s1] = action_space[np.argmax(R_list)]

            # print("Action: ",  pi[i_s0][i_s1], ", argmax: ", np.argmax(R_list))

    print("Done!")

    Utils.visualize(V, pi, state_space)

    # Print states, where we had no reward
    if STOCHASTIC:
        print("WARNING: We took the true model reward (because we could not"
              " estimate a good discrete reward) for: ")
        for rin in reward_is_none:
            print("State [{}, {}] in total {} times."
                  .format(rin[0], rin[1], rin[2]))

    Evaluation.evaluate(env, 100, discretize_index, pi, True)


if CALC_DISC_REWARDS:
    learn_discrete_rewards()

value_iteration(state_space, action_space, stochastic=STOCHASTIC, sigma=SIGMA)
