import gym
import quanser_robots
import numpy as np
from MemoryDQN import MemoryDQN
from DQNNet import DQN
import math
import random
import matplotlib.pyplot as plt


import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F


class ActionDisc(gym.Space):
    #self defined action space
    def __init__(self, high, low, number):
        gym.Space.__init__(self, (), np.float)
        self.high = high
        self.low = low
        self.n = number
        self.space = np.linspace(self.low, self.high, self.n)
        print(self.space)

    def sample(self):
        return random.choice(self.space)

    def contains(self, x):
        '''
        :param x: action space values (as array or similar)
        :return: index of the given actions (as list)
        '''
        indices = []
        for i in x:
            indices.append(np.where(self.space==i)[0][0])
        return indices


FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor


EPISODES = 200
BATCH_SIZE = 100
GAMMA = 0.9
HIDDEN_LAYER_NEURONS = 100
LEARNING_RATE = 0.001
ACTION_SPACE = 49

EPSILON = 0.05
env = gym.make("CartpoleStabShort-v0")
#env = gym.make("Pendulum-v0")

#define a new discrete action space
env.action_space =ActionDisc(env.action_space.high, env.action_space.low, ACTION_SPACE)

#creates the replay buffer and the neural network
memory = MemoryDQN(10000)
model = DQN(HIDDEN_LAYER_NEURONS, ACTION_SPACE, env.observation_space.shape[0])
#target = DQN(HIDDEN_LAYER_NEURONS, ACTION_SPACE, env.observation_space.shape[0])
#print(model.parameters)

optimizer = optim.Adam(model.parameters(), LEARNING_RATE)


cum_reward = []


def select_action(state_pred):
    #global steps_done
    sample = random.random()
    '''eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    eps_threshold = 0.2'''
    eps_threshold = EPSILON
    if sample > eps_threshold:
        with torch.no_grad():
            #change predicted states to torch tensor
            state_pred = torch.from_numpy(state_pred).type(FloatTensor).unsqueeze(0)
            #predict the actions to the given states
            pred_actions = model(Variable(state_pred))
            #find the action with the best q-value
            max_action = pred_actions.max(1)[1]
            #return the best action as tensor
            return torch.tensor(np.array([[env.action_space.space[max_action]]]))
    #exploration
    else:
        #return a random action of the action space
        return torch.tensor(np.array([[env.action_space.sample()]]))


for epi in range(EPISODES):
    cum_reward.append(0)
    state = env.reset()
    step = 0
    steps_done = 0
    loss = 100

    while True:
        action = select_action(state)
        state_follows, reward, done, info = env.step(action.numpy()[0])

        cum_reward[epi] += reward

        memory.add_observation(state, action, reward, state_follows)

        if epi == EPISODES - 1:
            env.render()

        #training
        if memory.size_mem() > BATCH_SIZE:
            states, actions, rewards, next_states = memory.random_batch(BATCH_SIZE)

            #find the index to the given action
            actions = env.action_space.contains(actions)
            #repeat it for the gather method of torch
            actions = np.array(actions).repeat(ACTION_SPACE).reshape(BATCH_SIZE,ACTION_SPACE)
            #change it to a long tensor (instead of a float tensor)
            actions = LongTensor(actions)

            #for each q-value(for each state in the batch and for each action), take the one from the chosen action
            current_q_values = model(states).gather(dim=1, index=actions)[:, 0]

            # neural net estimates the q-values for the next states, take the ones with the highest values
            max_next_q_values = model(next_states).detach().max(1)[0]
            expected_q_values = rewards + (GAMMA * max_next_q_values)

            #loss = F.smooth_l1_loss(current_q_values, expected_q_values.type(FloatTensor))
            loss = F.mse_loss(current_q_values, expected_q_values.type(FloatTensor))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = state_follows
        step += 1

        '''if done:
            break'''
        if step == 500:
            break
    print(cum_reward[-1], epi, loss)
plt.plot(cum_reward)
plt.show()