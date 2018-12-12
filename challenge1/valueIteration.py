from __future__ import print_function
import sys
import gym
import quanser_robots
import numpy as np
import matplotlib.pyplot as plt
from Regression import Regressor
from Utils import *
from Evaluation import *

env = gym.make('Pendulum-v2')
reg = Regressor()


# Returns discrete index of x in space
def get_index(space, x):
    index = []
    for i in range(len(x)):
        for a in range(len(space[i][:])-1):
            # TODO: this solution will not return anything for x = obs_space.high !!
            if space[i][a] <= x[i] < space[i][a+1]:
                index.append(a)
                break
            elif x[i] == env.observation_space.high[i]:
                # TODO: check if correct !!
                index.append(len(space[i][:])-1)
    return np.array(index)


def value_iteration(S, A, P, R, gamma, theta):

    print("Value iteration... ")

    S_shape = np.shape(S)
    A_shape = np.shape(A)

    V = np.zeros(shape=S_shape[0]*[S_shape[1]])
    PI = np.zeros(shape=S_shape[0]*[S_shape[1]])

    t = 1
    states = np.stack(np.meshgrid(range(len(S[0][:])),range(len(S[1][:])))).T.reshape(-1,2)
    while True:
        for s0,s1 in states:
            v = V[s0][s1]
            Qsa = np.zeros(shape=A_shape[0]*[A_shape[1]])
            for a in range(len(A[0][:])):
                next_state = reg.regressorState.predict(np.array([S[0][s0], S[1][s1], A[0][a]]).reshape(1, -1))[0]
                ns = get_index(space=S, x=next_state)
                ns0 = ns[0]
                ns1 = ns[1]
                Qsa[a] = P[s0][s1][a][ns0][ns1]*(R[ns0][ns1]+gamma*V[ns0][ns1])
            max_Qsa = np.max(Qsa)
            V[s0][s1] = max_Qsa
            delta = np.abs(max_Qsa-v)
        # Reduce discount factor per timestep
        #gamma = gamma/t
        t += 1
        print("Delta =", delta, end='')
        if(delta < theta):
            print(" ... done")
            break
        else:
            print(" < Theta =", theta)
    print()

    # Define policy
    print("Defining Policy ...", end='')
    sys.stdout.flush()
    for s0,s1 in states:
        Qsa = np.zeros(shape=A_shape[0] * [A_shape[1]])
        for a in range(len(A[0][:])):
            next_state = reg.regressorState.predict(np.array([S[0][s0], S[1][s1], A[0][a]]).reshape(1, -1))[0]
            ns = get_index(space=S, x=next_state)
            ns0 = ns[0]
            ns1 = ns[1]
            Qsa[a] = P[s0][s1][a][ns0][ns1] * (R[ns0][ns1] + gamma * V[ns0][ns1])
        # Get action for argmax index
        max_index = np.argmax(Qsa)
        max_action = A[0][max_index]
        PI[s0][s1] = max_action
    print("done")
    return V, PI


def evaluate_discrete_space(S, A):

    S_shape = np.shape(S)
    A_shape = np.shape(A)
    P = np.zeros(shape=(S_shape[0] * [S_shape[1]] +
                        A_shape[0] * [A_shape[1]] +
                        S_shape[0] * [S_shape[1]]))
    # R = np.zeros(shape=(S_shape[0] * [S_shape[1]] + A_shape[0] * [A_shape[1]]))

    # This reward 'function' is only defined for the successor states
    R = np.zeros(shape=(S_shape[0] * [S_shape[1]]))

    print("Evaluating reward function ... ", end='')
    sys.stdout.flush()
    # TODO: more efficient way?
    states = np.stack(np.meshgrid(range(R.shape[0]),range(R.shape[1]))).T.reshape(-1,2)

    for s0,s1 in states:
        R[s0][s1] = reg.regressorReward.predict(np.array([S[0][s0], S[1][s1]]).reshape(1, -1))
    print("done\n")

    print("Evaluating state transition function ... ", end='')
    sys.stdout.flush()
    # TODO: more efficent way?
    states_action = np.stack(np.meshgrid(range(P.shape[0]),range(P.shape[1]), range(P.shape[2]))).T.reshape(-1,3)

    for s0,s1,a in states_action:
        # Successor of state (s0, s1) for action a
        # We use [0] because we only have one state
        next_state = reg.regressorState.predict(np.array([S[0][s0], S[1][s1], A[0][a]]).reshape(1, -1))[0]
        # Get discrete index of next state
        ns = get_index(space=S, x=next_state)
        ns0 = ns[0]
        ns1 = ns[1]
        P[s0][s1][a][ns0][ns1] = 1
    print("done\n")

    return P, R


def build_discrete_space():

    """
        Creates discrete observation and action space
    :return:
    """

    state = env.reset()
    observation_size = len(state)
    observation_range = (env.observation_space.low, env.observation_space.high)
    # TODO: Different bins for different dimensions
    observation_bins = observation_size*[10]
    S = []
    for i in range(observation_size):
        S.append(np.linspace(observation_range[0][i], observation_range[1][i], observation_bins[i]))
    #print("Discrete Observation Space: ", S)


    action = env.action_space.sample()
    action_size = len(action)
    action_range = (env.action_space.low, env.action_space.high)
    action_bins = action_size*[10]
    A = []
    for i in range(action_size):
        A.append(np.linspace(action_range[0][i], action_range[1][i], action_bins[i]))
    #print("Discrete Action Space: ", A)


    return S, A


def evaluate(S, episodes, policy, render, sleep, epsilon_greedy=None):

    S_shape = np.shape(S)
    state_distribution = np.zeros(shape=S_shape[0]*[S_shape[1]])

    rewards_per_episode = []
    print("Evaluating...", end='')
    sys.stdout.flush()

    for e in range(episodes):

        state = env.reset()

        cumulative_reward = [0]

        for t in range(200):
            # Render environment
            if render:
                env.render()
                time.sleep(sleep)

            # discretize state
            index = get_index(S, state)
            state_distribution[index[0]][index[1]] += 1

            if epsilon_greedy is not None:
                rand = np.random.rand()
                if rand < epsilon_greedy:
                    action = np.random.uniform(low=env.action_space.low, high=env.action_space.high)
                else:
                    action = np.array([policy[index[0], index[1]]])
            else:
                # Do step according to policy and get observation and reward
                action = np.array([policy[index[0], index[1]]])

            next_state, reward, done, info = env.step(action)
            state = np.copy(next_state)

            cumulative_reward.append(cumulative_reward[-1] + reward)

            if done:
                print("Episode {} finished after {} timesteps".format(e + 1, t + 1))
                break

        rewards_per_episode.append(cumulative_reward)

    print("...done")

    # TODO: Look at calculation of mean cumulative rewards
    # Average reward over episodes
    rewards = np.average(rewards_per_episode, axis=0)

    env.close()

    # Plot rewards per timestep averaged over episodes
    plt.figure()
    plt.plot(rewards, label='Cumulative reward per timestep, averaged over {} episodes'.format(episodes))
    plt.legend()

    plt.figure()
    plt.imshow(state_distribution)
    plt.colorbar()
    plt.title("State Distribution")
    plt.show()


def main(make_plots):

    reg.perform_regression(env=env, epochs=10000, save_flag=False)
    S, A = build_discrete_space()
    P, R = evaluate_discrete_space(S=S, A=A)
    V, PI = value_iteration(S=S, A=A, P=P, R=R, gamma=0.9, theta=1e-2)
    evaluate(S=S, episodes=100, policy=PI, render=False, sleep=0, epsilon_greedy=0.1)

    if make_plots:
        visualize(value_function=V, policy=PI)

        plt.figure()
        plt.imshow(R)
        plt.title("Reward function")
        plt.show()

if __name__ == "__main__":
    main(make_plots=True)