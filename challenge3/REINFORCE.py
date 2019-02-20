import numpy as np
import gym
import quanser_robots
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K


class ReinforceAgent:
    def __init__(self, state_space_size, action_space, learning_rate, discounting, load_weights):
        self.state_space_size = state_space_size
        self.action_space = action_space
        self.load_weights_flag = load_weights
        self.gamma = discounting
        self.learning_rate = learning_rate
        self.model = self.init_model()
        self.model.summary()
        self.learning_rate = learning_rate
        self.trajectory_states, self.trajectory_actions, self.trajectory_probs, self.trajectory_rewards \
            = [], [], [], []
        if self.load_weights_flag:
            self.model.load_weights("./saved_models/REINFORCE")

    def init_model(self):
        """
        Creates the keras model of the nn policy approximation.
        :return: Keras model
        """
        model = Sequential()
        model.add(Dense(10, input_dim=self.state_space_size, activation='elu'))
        model.add(Dense(10, activation='elu'))
        model.add(Dense(self.action_space.size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, prob, reward):
        """
        Saving a trajectory.
        :param state: current state s of the environment
        :param action: current performed action a
        :param reward: current observed reward r
        :return: none
        """
        self.trajectory_states.append(state)
        self.trajectory_actions.append(action)
        self.trajectory_probs.append(prob)
        self.trajectory_rewards.append(reward)

    def predict_action(self, state):
        """
        Forward pass through the nn policy network.
        Approximating a ~ pi(a|s) and then random sampling from disc. action space.
        :param state: Current state s of the observation.
        :return: Predicted action a, probability of this action
        """
        state = state.reshape([1, state.shape[0]])
        action_probs = self.model.predict(state, batch_size=1).flatten()
        action = np.random.choice(self.action_space, 1, p=action_probs)[0]
        # return action, action_probs[np.where(self.action_space == action)]
        return action, action_probs

    def discounted_rewards(self, rewards):
        """
        Calculate discounted rewards from trajectory.
        :param rewards: Remembered rewards from trajectory (per time-step)
        :return:
        """
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self):
        """
        Calculate loss and weight update.
        Update parameters
        :return:
        """
        # Iterate over trajectory
        for t in range(0, len(self.trajectory_states)):
            # Get log gradient
            log_gradient = K.gradients(np.log(self.trajectory_probs[t]), self.model.trainable_weights)
            print("Log gradient:", log_gradient)
            # Update parameters
            self.model.set_weights(self.model.trainable_weights + self.learning_rate
                                   * log_gradient * self.discounted_rewards(self.trajectory_rewards[t:]))

        # Reset memory
        self.trajectory_states, self.trajectory_actions, self.trajectory_probs, self.trajectory_rewards =\
            [], [], [], []

    def load_weights(self, name):
        self.model.load_weights(name)

    def save_weights(self, name):
        self.model.save_weights(name)


def training(render=False):
    env = gym.make('Levitation-v1')
    print(env.spec.id)
    print("State Space:\t Shape: {}\t Low: {}\t High {}".format(env.observation_space.shape, env.observation_space.low,
                                                                env.observation_space.high))
    print("Action Space:\t Shape: {}\t Low: {}\t High {}".format(env.action_space.shape, env.action_space.low,
                                                                 env.action_space.high))
    print("Reward Range:\t {}".format(env.reward_range))
    discrete_actions = np.linspace(start=env.action_space.low, stop=env.action_space.high, num=5)
    print("Discrete Actions: ", discrete_actions)
    agent = ReinforceAgent(state_space_size=env.observation_space.shape[0], action_space=discrete_actions,
                           discounting=0.99, learning_rate=0.001, load_weights=False)

    epochs = 10
    for e in range(epochs):
        state = env.reset()
        cum_reward = [0]
        t = 0
        while True:
            if render:
                env.render()
            t += 1
            action, prob = agent.predict_action(state)
            action = np.array([action])
            # print("Action: ", action)
            state, reward, done, info = env.step(action)
            cum_reward.append(cum_reward[-1]+reward)
            agent.remember(state=state, action=action, prob=prob, reward=reward)
            if done:
                agent.train()
                print("Epoch {} done after {} time-steps. Cumulative reward: {}".format(
                    e, t, cum_reward[-1]
                ))
                break


if __name__ == "__main__":
    training()
