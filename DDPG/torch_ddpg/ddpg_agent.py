import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch_ddpg.neural_networks import Actor, Critic
from torch_ddpg.ActionNoise import OUNoise
from torch_ddpg.ReplayBuffer import ReplayBuffer


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, state_size, action_size, action_bounds, random_seed):
        """
        Initializes an DDPG Agent object.
        :param state_size: amount of dimensions of the environment's states
        :param action_size: amount of dimensions of the environment's actions
        :param action_bounds: Tuple(s) of bounds (low, high) for the action space
        :param random_seed: random seed for repeatability
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_bounds = action_bounds
        self.seed = random.seed(random_seed)

        # Actor Network and Actor Target Network
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network and Critic Target Network
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done):
        """
        Saves one step of experience (s, a, r, s', done) in the replay buffer.
        Then samples a random mini-batch from the replay buffer and learns from that sample.
        Only samples from the replay buffer, if it's size is bigger than the mini-batch size.
        :param state: current state of observation
        :param action: current action, chosen by the actor network
        :param reward: observed reward after performing action
        :param next_state: observed next state after performing action
        :param done: observed 'done' flag, inducing finished episodes
        :return: None
        """
        # Add experience to replay buffer
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            # action = self.actor_local(state).cpu().data.numpy()
            action = self.actor_local(state).cpu().data.numpy()*2  # TODO: Make this modular for all envs; *2 works for pendulum
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        # return np.clip(action, -1, 1)  # TODO: This should be clipped to action bounds instead
        return np.clip(action, self.action_bounds[0], self.action_bounds[1])

    def reset(self):
        """
        Resets the noise generator for the action exploration.
        :return: None
        """
        self.noise.reset()

    # TODO: understand how gradients are calculated and used
    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """
        Updates the target networks parameters, according to:
        θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_model: PyTorch model for actor/critic network
        :param target_model: PyTorch model for actor/critic target network
        :param tau: interpolation parameter
        :return: None
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
