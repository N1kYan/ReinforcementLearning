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
        indices = []
        for i in x:
            indices.append(np.where(self.space==i)[0][0])
        return indices


FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor


EPISODES = 200
BATCH_SIZE = 64
GAMMA = 0.9
HIDDEN_LAYER_NEURONS = 100
LEARNING_RATE = 0.001
ACTION_SPACE = 49 #49

EPS_END = 0.05
EPS_START = 0.9
EPS_DECAY = 200
env = gym.make("CartpoleStabShort-v0")
env.action_space =ActionDisc(env.action_space.high, env.action_space.low, ACTION_SPACE)

memory = MemoryDQN(1000)
model = DQN(HIDDEN_LAYER_NEURONS, ACTION_SPACE)

optimizer = optim.Adam(model.parameters(), LEARNING_RATE)


state = env.reset()
step = 0
steps_done = 0
action = np.ndarray([1,])
cum_reward = []
def select_action(state_pred):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest value for column of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            #state_pred = np.array(state_pred)
            #state_pred = FloatTensor(state_pred)
            state_pred = torch.from_numpy(state_pred).type(FloatTensor).unsqueeze(0)
            #print(model(Variable(state_pred)).max(1)[0].view(1,1))
            #print(env.action_space.space[model(Variable(state_pred)).max(1)[1].view(1,1)])
            #return model(Variable(state_pred)).max(1)[1].view(1,1)
            return torch.tensor(np.array([[env.action_space.space[model(Variable(state_pred)).max(1)[1].view(1,1)]]]))


    else:
        return torch.tensor(np.array([[env.action_space.sample()]]))


for epi in range(EPISODES):
    cum_reward.append(0)
    state = env.reset()
    step = 0
    steps_done = 0
    print(epi)
    while True:
        action = select_action(state)
        state_follows, reward, done, info = env.step(action.numpy()[0])

        cum_reward[epi] += reward

        memory.add_observation(state, action, reward, state_follows)

        if epi == EPISODES - 1:
            env.render()
        if done:
            reward = -1
        #training
        if memory.size_mem() > BATCH_SIZE:
            states, actions, rewards, next_states = memory.random_batch(BATCH_SIZE)

            states = torch.from_numpy(states).type(FloatTensor)

            next_states = torch.from_numpy(next_states).type(FloatTensor)

            #find the index to the given action
            actions = env.action_space.contains(actions)
            actions = np.array(actions).repeat(ACTION_SPACE).reshape(BATCH_SIZE,ACTION_SPACE)
            actions = LongTensor(actions)

            #for each q-value(for each state in the batch and for each action), take the one from the chosen action
            current_q_values = model(states).gather(dim=1, index=actions)[:, 0]

            # expected Q values are estimated from actions which gives maximum Q value
            max_next_q_values = model(next_states).detach().max(1)[0]
            expected_q_values = rewards + (GAMMA * max_next_q_values)

            loss = F.smooth_l1_loss(current_q_values, expected_q_values.type(FloatTensor))
            #loss = F.mse_loss(current_q_values, expected_q_values.type(FloatTensor))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = state_follows
        step += 1

        '''if done:
            break'''
        if step == 500:
            break

plt.plot(cum_reward)
plt.show()