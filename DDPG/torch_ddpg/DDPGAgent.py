import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch_ddpg.NeuralNetworks import Actor, Critic
from torch_ddpg.ReplayBuffer import ReplayBuffer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, state_size, action_size, action_bounds, random_seed, buffer_size, batch_size, gamma, tau,
                 lr_actor, lr_critic, weight_decay, noise_generator):
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
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.TAU = tau
        self.LR_ACTOR = lr_actor
        self.LR_CRITIC = lr_critic
        self.WEIGHT_DECAY = weight_decay

        # Actor Network and Actor Target Network
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.LR_ACTOR)

        # Critic Network and Critic Target Network
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.LR_CRITIC,
                                           weight_decay=self.WEIGHT_DECAY)

        # Noise process
        self.noise = noise_generator
        # self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.BUFFER_SIZE, self.BATCH_SIZE, random_seed)
    
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
        :param perform_update: If true learn from sample; Otherwise only store experience
        :return: None
        """
        # Add experience to replay buffer
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, self.GAMMA)

    def act(self, state, add_noise=True):
        """
        Outputs action from actor network / current policy given a state.
        :param state: Current state of the environment
        :param add_noise: Boolean for adding noise or not
        :return: The output action from the actor network
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():

            # Activation function tanh returns [-1, 1] so we multiply by
            # the highest possible action to map it to our action space.
            action = self.actor_local(state).cpu().data.numpy()\
                     * self.action_bounds[1]

        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        # Clip actions to action bounds (low, high)
        return np.clip(action, self.action_bounds[0], self.action_bounds[1])

    def reset(self):
        """
        Resets the noise generator for the action exploration.
        :return: None
        """
        self.noise.reset()

    def learn(self, experiences, gamma):
        """
        Updates the actor(policy) and critic(value function) networks' parameters
        given a radnom mini-batch of experience samples from the replay buffer.

            Q_targets = r + γ * critic_target(next_state, actor_target(next_state))

        :param experiences: Mini-batch of random samples for the replay buffer
        :param gamma: Discount factor
        :return: None
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
        critic_loss = F.mse_loss(Q_expected, Q_targets)  # REG ERROR

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
        self.soft_update(self.critic_local, self.critic_target, self.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.TAU)

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
