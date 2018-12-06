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
        for s in range(env.state_space_shape):
            Q_sa = np.zeros(env.action_space_shape)
            for a in range(env.action_space_shape):
                for prob_s, next_state, reward, _ in env.P[s][a]:
                    Q_sa[a] += prob_s * (reward + gamma * value_function[next_state])
            maxi = np.max(Q_sa) # TODO: ??
            delta = np.max(delta, np.abs(maxi-value_function[s]))
            value_function[s] = maxi
        if delta < theta:
            break

    # Initialize policy
    policy = np.zeros([env.state_space_shape, env.action_space_shape])

    # Iterate to converge to optimal policy
    for s in range(env.state_space_shape):
        Q_sa = np.zeros(env.action_space_shape)
        for a in range(env.action_space_shape):
            for prob_s, next_state, reward, _ in env.P[s][a]:
                Q_sa[a] += prob_s * (reward + gamma * value_function[next_state])
        best_action = np.argmax(Q_sa)
        policy[s] = np.eye(env.action_space_shape)[best_action]

    return value_function, policy


def main():
    env = gym.make('Pendulum-v2')
    disc_env = DiscreteEnvironment(env=env, name='EasyPendulum',
                                   state_space_shape=(16+1, 16+1),
                                   action_space_shape=(16+1,))
    disc_env.evaluate_transition_prob(env=env, epochs = 10000, save_flag=False) # TODO
    value_function, policy = value_iteration(env=disc_env, theta=1e-1, gamma=0.1)


if __name__=="__main__":
    main()