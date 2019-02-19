import numpy as np
import gym
import quanser_robots
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K


class ReinforceAgent:
    def __init__(self, state_space_size, action_space_size, learning_rate, discounting):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.render_flag = False
        self.load_weights_flag = False
        self.gamma = discounting
        self.learning_rate = learning_rate
        self.model = self.init_model()
        self.model.summary()
        self.learning_rate = learning_rate
        self.states, self.gradients, self.rewards, self.probs = [], [], [], []
        if self.load_weights_flag:
            self.model.load_weights("./saved_models/REINFORCE")

    def init_model(self):
        """
        Creates the keras model of the nn policy approximation.
        :return: Keras model
        """
        model = Sequential()
        model.add(Dense(20, input_dim=self.state_space_size, activation='elu'))
        model.add(Dense(20, activation='elu'))
        model.add(Dense(self.action_space_size, activation='softmax'))
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_space_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)

    def predict_action(self, state):
        """
        Forward pas through the nn policy network.
        Approximating a ~ pi(a|s)
        :param state: Current state s of the observation.
        :return: Predicted action a
        """
        state = state.reshape([1, state.shape[0]])
        action_probs = self.model.predict(state, batch_size=1).flatten()
        self.probs.append(action_probs)
        probs = action_probs/np.sum(action_probs)
        action = np.random.choice(self.action_space_size, 1, p=probs)[0]
        return action, probs

    def discounted_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self):
        gradients = np.vstack(self.gradients)
        rewards = self.discounted_rewards(np.vstack(self.rewards))
        rewards = rewards / np.std(rewards - np.mean(rewards))
        gradients *= rewards

        X = np.squeeze(np.vstack([self.states]))
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        self.model.train_on_batch(X, Y)
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []

    def load_weights(self, name):
        self.model.load_weights(name)

    def save_weights(self, name):
        self.model.save_weights(name)


def training():
    env = gym.make('Levitation-v1')
    print(env.spec.id)
    print("State Space:\t Shape: {}\t Low: {}\t High {}".format(env.observation_space.shape, env.observation_space.low,
                                                                env.observation_space.high))
    print("Action Space:\t Shape: {}\t Low: {}\t High {}".format(env.action_space.shape, env.action_space.low,
                                                                 env.action_space.high))
    print("Reward range:\t {}".format(env.reward_range))
    agent = ReinforceAgent(state_space_size=env.observation_space.shape[0], action_space_size=env.action_space.shape[0],
                           discounting=0.99, learning_rate=0.001)
    epochs = 10
    render = True
    for e in range(epochs):
        state = env.reset()
        cum_reward = [0]
        t = 0
        while True:
            t += 1
            action, prob = agent.predict_action(state)
            action = np.array([action])
            # print("Action: ", action)
            state, reward, done, info = env.step(action)
            cum_reward.append(cum_reward[-1]+reward)
            agent.remember(state, action, prob, reward)
            if done:
                agent.train()
                print("Epoch {} done after {} time-steps. Cumulative reward: {}".format(
                    e, t, cum_reward[-1]
                ))
                break


if __name__ == "__main__":
    training()
