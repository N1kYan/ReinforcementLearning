import numpy as np
import gym
import quanser_robots
from DiscreteEnvironment import DiscreteEnvironment

# TODO: comments


def value_iteration(env, theta, gamma):

    # Initialize value function
    value_function = np.zeros(env.state_space_shape)

    # Iterate to converge to optimal value function
    while True:
        delta = 0
        for s0 in range(env.state_space_shape[0]):
            for s1 in range(env.state_space_shape[1]):
                Q_sa = np.zeros(env.action_space_shape)
                for a in range(env.action_space_shape[0]):
                    for prob_s, next_state, reward, _ in env.P[s0][s1][a]:
                        Q_sa[a] += prob_s * (reward + gamma * value_function[next_state[0]][next_state[1]])
                maxi = np.max(Q_sa)
                delta = np.max(delta, np.abs(maxi-value_function[s0][s1]))
                value_function[s0][s1] = maxi
        if delta < theta:
            break

    # Initialize policy
    policy = np.zeros([env.state_space_shape[0], env.state_space_shape[1], env.action_space_shape])

    # Iterate to converge to optimal policy
    for s0 in range(env.state_space_shape[0]):
        for s1 in range(env.state_space_shape[1]):
            Q_sa = np.zeros(env.action_space_shape)
            for a in range(env.action_space_shape[0]):
                for prob_s, next_state, reward, _ in env.P[s0][s1][a]:
                    Q_sa[a] += prob_s * (reward + gamma * value_function[next_state[0], next_state[1]])
            best_action = np.argmax(Q_sa)
            policy[s0][s1] = np.eye(env.action_space_shape)[best_action]

        return value_function, policy


def main():
    env = gym.make('Pendulum-v2')
    disc_env = DiscreteEnvironment(env=env, name='EasyPendulum',
                                   state_space_shape=(8+1, 8+1),
                                   action_space_shape=(8+1,))
    state = env.reset()
    disc_env.perform_regression(env=env, epochs=10000, save_flag=False)

    # Covariance matrix instead of sigma list??
    disc_env.get_successors(state=state, action=[1.0], sigmas=[.1, .1])
    # value_function, policy = value_iteration(env=disc_env, theta=1e-1, gamma=0.1)


if __name__ == "__main__":
    main()