import gym
import quanser_robots
import numpy as np
import matplotlib.pyplot as plt
from MemoryDQN import MemoryDQN
from DQNNet import DQN
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


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
GAMMA = 0.8
HIDDEN_LAYER_NEURONS = 100
LEARNING_RATE = 0.001
ACTION_SPACE = 49

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

    while True:
        act = select_action(state)
        #action[0]=act[0].numpy()
        action = act
        #act = action.numpy()
        print(action.numpy())
        state_follows, reward, done, info = env.step(act.numpy()[0])
        #state_follows = np.concatenate(state_follows)
        #memory.add_observation(FloatTensor([state]), action, FloatTensor([reward]), FloatTensor([state_follows]))
        memory.add_observation(state, action, reward, state_follows)
        env.render()
        if epi == EPISODES - 1:
            env.render()
        if done:
            reward=-1
        #training
        if memory.size_mem()> BATCH_SIZE:
            states, actions, rewards, next_states = memory.random_batch(BATCH_SIZE)
            # current Q values are estimated by NN for all actions

            states = torch.from_numpy(states).type(FloatTensor)
            next_states = torch.from_numpy(next_states).type(FloatTensor)

            #actions = torch.from_numpy(actions).type(FloatTensor).unsqueeze(0)
            actions = env.action_space.contains(actions)
            actions = LongTensor(actions)
            #print(np.shape(states), np.shape(actions))
            current_q_values = model(states).gather(dim=1, index=actions)
            # expected Q values are estimated from actions which gives maximum Q value
            max_next_q_values = model(next_states).detach().max(1)[0]
            expected_q_values = rewards + (GAMMA * max_next_q_values)
            print(expected_q_values)
            # loss is measured from error between current and newly expected Q values
            loss = F.smooth_l1_loss(current_q_values, expected_q_values)

            # backpropagation of loss to NN
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        state = state_follows
        step += 1

        if done:
            break