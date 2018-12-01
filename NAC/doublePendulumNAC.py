import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from quanser_robots import GentlyTerminating
from quanser_robots.double_pendulum.examples import metronom



def initialize_objects():
    # Policy network
    policyNetworkModel = Sequential()
    policyNetworkModel.add(Dense(12, input_dim=6, init='uniform', activation='relu'))
    policyNetworkModel.add(Dense(12, init='uniform', activation='relu'))
    policyNetworkModel.add(Dense(1, init='uniform', activation='tanh'))
    policyNetworkModel.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return policyNetworkModel

def draw_initial_state(env):
    return 0 #TODO: definie inital state distribution modular for possible environments

# TODO: How to calculate policy network gradient? Should return np.array
def train(env, policy, policy_der, sigma):

    # Initial state and parameters
    state = draw_initial_state(env)
    A = 0
    z = 0
    b = z

    for t in range (100):
        # Draw action from policy network and observe new state and reward
        action = policy(state)
        new_state, reward, done, infos = env.step(action)

        # Critict Evaluation

        # Update basis functions
        sigma_tilde = np.array([sigma(new_state).T, 0]).T
        sigma_hat = np.array([sigma(state).T, policy_der(state).T]).T

        # Statistics
        # TODO: Define delta and gamma; What are their meanings?
        z = delta*z + sigma_hat
        A = A + z*(sigma_hat-gamma*sigma_tilde).T
        b = b + z*reward

        # Critic parameters
        vec = (np.invert(A)*b).T
        w = vec[0].T
        v = vec[1].T

        # Actor update
        # TODO: Define alpha and beta; What are their meanings?
        theta = theta = alpha*w
        z = beta*z*A
    return policy

def evaluate(env, crtl):

    obs = env.reset()

    while True:
        env.render()
        if crtl is None:
            act = env.action_space.sample()
        else:
            act = np.array(crtl(obs))
        obs, rwd, done, info = env.step(act)

    env.close()

def main():
    env = GentlyTerminating(gym.make('DoublePendulum-v0'))
    """
        The DoublePendulum-v0 environment:

        The state space is 6 dimensional:
         (x, theta1, theta2, x dot, theta1 dot, theta2 dot)
        Min: (-2, -pi, -30, -40) Max: (2, pi, 30, 40)

        The action space is 1 dimensional.
        Min -15 Max: 15
    """

    ctrl = ...  # some function f: s -> a
    obs = env.reset()

    policyNetworkModel = initialize_objects()
    #ctrl = metronom.MetronomCtrl()
    ctrl = train(policyNetworkModel)
    evaluate(env=env, ctrl=None)

if __name__ == "__main__":
    main()