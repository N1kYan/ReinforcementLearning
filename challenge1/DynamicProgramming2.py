import numpy as np
import TrueModel
import gym
import Utils
import sys
import Evaluation

"""
    Alex wants to implement value iteration again  which is  why he created  
    this class.
"""


# PARAMETER
STATE_SPACE_SIZE = (50 + 1, 50 + 1)
ACTION_SPACE_SIZE = (16 + 1)
THETA = 0.1
GAMMA = 0.8

# Variables
env = gym.make('Pendulum-v2') # Create gym/quanser environment
action_space = (np.linspace(env.action_space.low, env.action_space.high,
                            ACTION_SPACE_SIZE))
state_space = (np.linspace(-np.pi, np.pi, STATE_SPACE_SIZE[0]),
               np.linspace(-8, 8, STATE_SPACE_SIZE[1]))



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


def stochastic_reward(state_space, main_probability, s_dash,
                                  immediate_reward, V):
    exp_reward = 0
    index = discretize_index(s_dash)
    s_indices, probs = get_probs(state_space, s_dash, 1)

    #iterate over all possible states
    for i, s_index in enumerate(s_indices):
        V_dash = V[s_index, index[1]]
        # print(V_dash)
        R = immediate_reward + (GAMMA * V_dash)
        # print(R)
        # print(probs[i])
        exp_reward += probs[i] * R

    # sys.exit()
    return exp_reward


def stochastic_reward_3states(state_space, main_probability, index,
                                  immediate_reward, V):

    exp_reward = 0
    V_dash = V[index[0], index[1]]

    # -200 is discounted and therefore gets better
    # 0.8 * -200 = -160
    R = immediate_reward + (GAMMA * V_dash)
    exp_reward += main_probability * R

    index[0] = (index[0]+1) % len(state_space[0])
    V_dash = V[index[0], index[1]]
    R = immediate_reward + (GAMMA * V_dash)
    exp_reward += (1-main_probability)/2 * R

    index[0] = (index[0]-2) % len(state_space[0])
    V_dash = V[index[0], index[1]]
    R = immediate_reward + (GAMMA * V_dash)
    exp_reward += (1-main_probability)/2 * R

    return exp_reward



def value_iteration(state_space, action_space):

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
                    # print("next state: ", s_dash)
                    immediate_reward = TrueModel.reward([s0, s1, a])

                    R = stochastic_reward(state_space, 0.4, s_dash,
                                          immediate_reward, V)

                    # print(R)

                    # ENTWEDER ODER

                    # index = discretize_index(s_dash)
                    # V_dash = V[index[0], index[1]]
                    # R = immediate_reward + (GAMMA * V_dash)

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
                index = discretize_index(s_dash)
                V_dash = V[index[0], index[1]]

                print("-- ", i_s0, ", ", i_s1, ", ", a, " --")
                print("s0: ", s0, ", s1: ", s1, ", a: ", a)
                print("s_dash: ", s_dash, " index: ", index, " V_dash: ", V_dash)

                R = TrueModel.reward([s0, s1, a]) + (GAMMA * V_dash)
                R_list.append(R)




            print(i_s0, i_s1, R_list)

            pi[i_s0][i_s1] = action_space[np.argmax(R_list)]

            print("Action: ",  pi[i_s0][i_s1], ", argmax: ", np.argmax(R_list))

    print("Done!")

    Utils.visualize(V, pi, state_space)

    Evaluation.evaluate(env, 100, discretize_index, pi, True)

value_iteration(state_space, action_space)

