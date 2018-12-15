from __future__ import print_function
import sys
import gym
import quanser_robots
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from Regression import Regressor
from Utils import *
from Evaluation import *

env = gym.make('Pendulum-v2')
reg = Regressor()


# Returns discrete index of x in space
def get_index(space, x):
    index = []
    for dim in range(len(x)):
        for ind in range(len(space[dim][:])-1):
            if space[dim][ind] <= x[dim] < space[dim][ind+1]:
                index.append(ind)
                break
            elif x[dim] == env.observation_space.high[dim]:
                # TODO: check if correct !!
                index.append(len(space[dim][:])-2)
    return np.array(index)


def value_iteration(S, A, P, R, gamma, theta):

    print("Value iteration... ")

    # Initialize value function with bad values
    V = np.zeros(shape=np.shape(S)[0]*[np.shape(S)[1]-1])
    V.fill(-20)
    goal = get_index(S, [0, 0])
    V[goal[0]][goal[1]] = 0
    # Fill policy with neutral action (mid of action space)
    PI = np.zeros(shape=np.shape(S)[0]*[np.shape(S)[1]-1])
    PI.fill(A[0][int(len(A[0][:])/2)])
    print(A[0][int(len(A[0][:])/2)])

    t = 1
    states = np.stack(np.meshgrid(range(len(S[0][:])-1),range(len(S[1][:])-1))).T.reshape(-1, 2)
    while True:
        for s0, s1 in states:
            v = V[s0][s1]
            Qsa = np.zeros(shape=np.shape(A)[0]*[np.shape(A)[1]-1])
            for a in range(len(A[0][:])-1):
                next_state = reg.regressorState.predict(np.array([S[0][s0], S[1][s1], A[0][a]]).reshape(1, -1))[0]
                ns = get_index(space=S, x=next_state)
                ns0 = ns[0]
                ns1 = ns[1]
                Qsa[a] = P[s0][s1][a][ns0][ns1]*(R[ns0][ns1]+gamma*V[ns0][ns1])
            max_Qsa = np.max(Qsa)
            V[s0][s1] = max_Qsa
            delta = np.abs(v-max_Qsa)
        # Reduce discount factor per timestep
        #gamma = gamma/t
        t += 1
        print("Delta =", delta, end='')
        if delta < theta:
            print(" ... done")
            break
        else:
            print(" > Theta =", theta)
    print()

    # Define policy
    print("Defining Policy ...", end='')
    sys.stdout.flush()
    for s0, s1 in states:
        Qsa = np.zeros(shape=np.shape(A)[0] * [np.shape(A)[1]-1])
        for a in range(len(A[0][:])-1):
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


def evaluate_discrete_space(S, A, gaussian_sigmas):

    P = np.zeros(shape=(np.shape(S)[0] * [np.shape(S)[1]-1] +
                        np.shape(A)[0] * [np.shape(A)[1]-1] +
                        np.shape(S)[0] * [np.shape(S)[1]-1]))
    # R = np.zeros(shape=(S_shape[0] * [S_shape[1]] + A_shape[0] * [A_shape[1]]))

    # This reward 'function' is only defined for the successor states
    R = np.zeros(shape=(np.shape(S)[0] * [np.shape(S)[1]-1]))

    # This part is defining the value function by predicting the reward of every discrete
    # state action pair
    # TODO: more efficient way?
    print("Evaluating reward function ... ", end='')
    sys.stdout.flush()
    states = np.stack(np.meshgrid(range(R.shape[0]),range(R.shape[1]))).T.reshape(-1,2)
    for s0, s1 in states:
        R[s0][s1] = reg.regressorReward.predict(np.array([S[0][s0], S[1][s1]]).reshape(1, -1))
    print("done\n")

    # This part is defining the state transition prob. by predicting the resulting state
    # for every state action pair
    # TODO: more efficent way?
    print("Evaluating state transition function ... ", end='')
    sys.stdout.flush()
    states_action = np.stack(np.meshgrid(range(P.shape[0]),range(P.shape[1]), range(P.shape[2]))).T.reshape(-1, 3)
    for s0, s1, a in states_action:
        # Successor of state (s0, s1) for action a
        # We use [0] because we only have one state
        next_state = reg.regressorState.predict(np.array([S[0][s0], S[1][s1], A[0][a]]).reshape(1, -1))[0]
        # Get discrete index of next state
        ns = get_index(space=S, x=next_state)
        ns0 = ns[0]
        ns1 = ns[1]

        #main_p = 1.0
        #P[s0][s1][a][ns0][ns1] = main_p

        successor_indices = np.stack(np.meshgrid(range(P.shape[3]), range(P.shape[4]))).T.reshape(-1, 2)

        cov = np.eye(2, 2)
        cov[0][0] = gaussian_sigmas[0]
        cov[0][1] = 0
        cov[0][1] = 0
        cov[1][1] = gaussian_sigmas[1]

        max_index0 = np.shape(P)[0]

        for i0, i1 in successor_indices:
            P[s0][s1][a][i0][i1] = 0.5 * (multivariate_normal.pdf(x=np.array([i0, i1]),
                                                           mean=np.array([ns0, ns1]), cov=cov)
                                          + multivariate_normal.pdf(x=np.array([i0, i1]),
                                                            mean=np.array([max_index0+ns0, ns1]), cov=cov))
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
    observation_bins = observation_size*[20]
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

    state_distribution = np.zeros(shape=np.shape(S)[0]*[np.shape(S)[1]-1])

    rewards_per_episode = []
    print("Evaluating...")
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
    plt.show()

    return state_distribution


def main(make_plots):

    reg.perform_regression(env=env, epochs=10000, save_flag=False)
    S, A = build_discrete_space()
    P, R = evaluate_discrete_space(S=S, A=A, gaussian_sigmas=np.array([1, 1]))

    plt.figure()
    plt.imshow(P[1][1][1][:][:])
    plt.title("State transition probability for S:1|1 A:1")
    plt.colorbar()
    plt.grid()

    V, PI = value_iteration(S=S, A=A, P=P, R=R, gamma=0.95, theta=1e-5)
    state_distribution = evaluate(S=S, episodes=10, policy=PI, render=False, sleep=0, epsilon_greedy=0.01)

    if make_plots:
        visualize(value_function=V, policy=PI, R=R, state_distribution=state_distribution, state_space=S)


if __name__ == "__main__":
    main(make_plots=True)