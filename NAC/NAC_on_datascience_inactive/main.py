import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import quanser_robots
import sys
#from quanser_robots import GentlyTerminating
#from quanser_robots.double_pendulum.examples import metronom


# TODO
def pi(state, theta):
    pass


# TODO
def pi_gradient(state, theta):
    pass


def nac_with_lstd(env, epochs, phis, theta_initial, w_initial, alpha, beta, gamma, delta, epsilon):
    # Draw initial state
    state = env.reset()

    # Initialize parameters
    A = np.zeros(shape=(0, 0))
    b = np.zeros(shape=(0, 0))
    z = np.zeros(shape=(0, 0))
    w = np.copy(w_initial)
    theta = np.copy(theta_initial)

    for t in range(epochs):
        env.render()
        # Draw action from parameterized policy
        action = pi(state, theta)
        next_state, reward, done, info = env.step(action)
        """
            Critic evaluation
            
        """
        e = 0
        while True:
            e += 1
            # Update basis functions
            phis_tilde = np.array([phis(next_state, w).T, np.zeros(shape=np.shape(phis(next_state, w))).T]).T
            phis_hat = np.array([phis(state, w).T, pi_gradient(state, theta).T]).T
            # Update statistics
            z = delta * z + phis_hat
            A = A + np.matmul(z, (phis - gamma * phis_tilde).T)
            b = b + z * reward
            # Observe critic parameters
            critic_params = np.matmul(np.linalg.inv(A), b)
            v = critic_params.T[0].T
            w_new = critic_params.T[1].T
            # Check for convergence
            if np.linalg.norm((w_new - w)) < epsilon:
                print("LSTD-Q finished after {} timesteps".format(e))
                break
        """
            Actor update
            
        """
        # Update policy parameters
        theta = theta + alpha * w_new
        # Forget sufficient statistics
        z = beta * z
        A = beta * A
        b = beta * b

        if done:
            print("Epoch finished after {} timesteps\n".format(t))
            break


def main():
    # env = GentlyTerminating(gym.make('DoublePendulum-v0'))
    """
        The DoublePendulum-v0 environment:

        The state space is 6 dimensional:
         (x, theta1, theta2, x dot, theta1 dot, theta2 dot)
        Min: (-2, -pi, -30, -40) Max: (2, pi, 30, 40)

        The action space is 1 dimensional.
        Min -15 Max: 15

    """
    # env = gym.make("DoublePendulum-v0')

    """
        The CartPole-v0 environment:
    """
    env = gym.make('CartPole-v0')

    # episodic_nac(env=env, sess=sess, updates=1, epochs=100, actor=actor, critic=critic, alpha=0.5, gamma=0.8)

    phis = ...

    w_initial = np.ones(shape=(len(phis), 1))

    theta_initial = ...

    nac_with_lstd(env=env, epochs=1000, phis=phis, theta_initial=..., w_initial=w_initial,
                  alpha=0.1, beta=0.0, gamma=0.5, delta=1.0, epsilon=1e-2)

    # sess.close()


if __name__ == "__main__":
    main()
