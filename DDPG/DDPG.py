import gym
import quanser_robots
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time
import datetime
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from NeuralNetworks import Actor, Critic
from ReplayBuffer import ReplayBuffer
from ActionNoise import OUNoise, Gaussian

# Use cuda if available
print("Torch Cuda available: {}".format(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluation(actor, epochs, render):
    """
    Evaluates the trained agent on the environment (both declaired above).
    Plots the reward per timestep for every episode.
    :param actor: Path leading to saved pytorch actor object
    :param epochs: Episodes for evaluation
    :param render: Set true to render the evaluation episodes
    :return: None
    """
    print("\nEvaluating ...")

    plt.figure()
    plt.title("Rewards during evaluation")
    plt.xlabel("Time-step")
    plt.ylabel("Current reward")

    all_rewards = []
    for e in range(1, epochs + 1):
        state = env.reset()
        rewards = []
        t = 0
        while True:
            t += 1
            if render:
                env.render()

            # Output action from actor network / current policy given the current state
            state = torch.from_numpy(state).float().to(device)
            with torch.no_grad():
                # Activation function tanh returns [-1, 1] so we multiply by
                # the highest possible action to map it to our action space.
                action = actor(state).cpu().data.numpy() * env_specs[3]
            # Clip actions to action bounds (low, high)
            action = np.clip(action, env_specs[2], env_specs[3])

            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            state = np.copy(next_state)
            if done:
                break
        all_rewards.append(np.array(rewards))
        env.close()
    for r in all_rewards:
        plt.plot(r)
    print("... done!")
    print("Average cumulative reward:", np.average([np.sum(r) for r in all_rewards]))
    plt.show()


def training(epochs, max_steps, epoch_checkpoint, noise, epsilon, epsilon_decrease, add_noise, lr_actor, lr_critic, weight_decay, memory,
             gamma, tau, seed, save_flag, load_flag, load_path, render):
    """
    Creates neural networks and runs the training process on the gym environment.
    Then plots the cumulative reward per episode.
    Saves actor and critic torch model.
    :param epochs: Number of epochs for training
    :param max_steps: Maximum time-steps for each training epoch;
     Does end epochs for environments, which epochs are not time limited
    :param epoch_checkpoint: Checkpoint for printing the learning progress and rendering the environment
    :param noise: The noise generating process; added to the action output of the actor network
    :param epsilon: Set different from None to use epsilon-greedy action selection
    :param epsilon_decrease: Set value to not none to multiply epsilon by epsilon_decrease every epoch checkpoint
    :param add_noise: Set true to add noise to enable action exploration
    :param lr_actor: Learning rate for actor network
    :param lr_critic: Learning rate for critic network
    :param weight_decay: Weight decay for critic network
    :param memory: The Replay Buffer object
    :param gamma: Discount factor for DDPG Learning
    :param tau: Parameter for 'soft' target updates
    :param seed: Random seed for repeatability
    :param save_flag: Set true to save the models, plots and parameters in text file; otherwise display plots directly
    :param load_flag: Set true to load pretrained model instead of training a new one
    :param load_path: Path to load pretrained model from
    :param render: Set true for rendering every 'epoch_checkpoint' episode
    :return: None
    """

    if load_flag:
        loaded_actor = Actor(state_size=env_specs[0], action_size=env_specs[1], seed=seed).to(device)
        savepoint = torch.load('./{}/{}'.format(env.spec.id, load_path))
        loaded_actor.load_state_dict(savepoint)
        return loaded_actor

    def learn(experiences):
        """
        DDPG learning rule (from paper).
        Updates the actor(policy) and critic(value function) networks' parameters
        given a radnom mini-batch of experience samples from the replay buffer.

            Q_targets = r + γ * critic_target(next_state, actor_target(next_state))

        :param experiences: Mini-batch of random samples for the replay buffer
        :return: None
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = actor_target(next_states)
        Q_targets_next = critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)  # REG ERROR

        # Minimize the loss
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = actor_local(states)
        # Gradient ascent to get actor parameters maximizing critic estimation of value function
        actor_loss = -critic_local(states, actions_pred).mean()
        # actor_loss = critic_local(states, actions_pred).mean()
        # Minimize the loss
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        """
        Updates the target networks parameters, according to:
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(actor_target.parameters(), actor_local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        for target_param, local_param in zip(critic_target.parameters(), critic_local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    # Create Actor and Critic network and their target networks
    actor_local = Actor(state_size=env_specs[0], action_size=env_specs[1], seed=seed).to(device)
    actor_target = Actor(state_size=env_specs[0], action_size=env_specs[1], seed=seed).to(device)
    actor_optimizer = optim.Adam(actor_local.parameters(), lr=lr_actor)

    # Critic Network and Critic Target Network
    critic_local = Critic(state_size=env_specs[0], action_size=env_specs[1], seed=seed).to(device)
    critic_target = Critic(state_size=env_specs[0], action_size=env_specs[1], seed=seed).to(device)
    critic_optimizer = optim.Adam(critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)

    # Reset agent's noise generator
    noise.reset()

    # Measure the time we need to learn
    time_start = time.time()

    # ----------------------- Main Training Loop ----------------------- #
    scores_deque = deque(maxlen=epoch_checkpoint)
    average_episode_reward = []
    cumulative_episode_reward = []
    for e in range(1, epochs + 1):
        state = env.reset()
        # noise.reset()
        cumulative_reward = 0
        episode_rewards = []
        t = 0
        for t_i in range(max_steps):
            t += 1
            if e % epoch_checkpoint == 0:
                # Render the 'epoch_checkpoint'
                if render:
                    env.render()
            # Output action from actor network / current policy given the current state
            state = torch.from_numpy(state).float().to(device)
            actor_local.eval()
            # Activation function tanh returns [-1, 1] so we multiply by
            # the highest possible action to map it to our action space.
            with torch.no_grad():
                if epsilon is not None:
                    rand = np.random.rand()
                    if rand > epsilon:
                        action = actor_local(state).cpu().data.numpy() * env_specs[3]
                    else:
                        action = env.action_space.sample()
                else:
                    action = actor_local(state).cpu().data.numpy() * env_specs[3]

            actor_local.train()
            if add_noise:
                action += noise.sample()
            # Clip actions to action bounds (low, high)
            # TODO: Clipping action after applying noise problematic?
            # https: // www.reddit.com / r / reinforcementlearning / comments / 8hgdad / ideas_for_exploration_noise_other_than_ornstein /
            action = np.clip(action, env_specs[2], env_specs[3])
            # print(action)
            # Perform the action
            next_state, reward, done, _ = env.step(action)
            episode_rewards.append(reward)

            # Add experience to replay buffer
            memory.add(state, action, reward, next_state, done)

            # Learn, if enough samples are available in memory
            if len(memory) > memory.batch_size:
                experience_sample = memory.sample()
                learn(experience_sample)

            state = next_state
            cumulative_reward += reward
            if done:
                cumulative_episode_reward.append(cumulative_reward)
                break
        env.close()
        scores_deque.append(cumulative_reward)
        average_episode_reward.append(np.mean(episode_rewards))
        print('\rEpisode {}\tAverage Reward: {}\tSteps: {}\tSigma: {}\t({:.2f} min elapsed)'.
              format(e, np.mean(scores_deque), t, noise.sigma, (time.time() - time_start) / 60), end="")
        if e % epoch_checkpoint == 0:
            # Decrease epsilon (percentage of random actions) every epoch checkpoint
            if epsilon is not None and epsilon_decrease is not None:
                epsilon = epsilon * epsilon_decrease
            noise.sigma = noise.sigma - 0.1
            # Print cumulative reward per episode averaged over #epoch_checkpoint episodes
            print('\rEpisode {}\tAverage Reward: {:.3f}\tSigma: {}\t({:.2f} min elapsed)'.
                  format(e, np.mean(scores_deque), noise.sigma, (time.time() - time_start) / 60))

    # ----------------------- Plotting, Saving, etc. ----------------------- #

    # Get current time
    save_time = datetime.datetime.now()

    # Create folder from current time
    try:
        os.makedirs("{}/{}-{}-{}".format(env.spec.id, save_time.day, save_time.month, save_time.hour))
    except FileExistsError:
        pass

    # Plot/Save average reward per episode
    plt.figure()
    plt.title("Average reward per episode")
    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    plt.plot(average_episode_reward)
    if save_flag:
        plt.savefig("./{}/{}-{}-{}/avg_reward.png".format(env.spec.id, save_time.day, save_time.month, save_time.hour))
        plt.close()

    # Plot/Save cumulative reward per episode
    plt.figure()
    plt.title("Cumulative reward per episode")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.plot(cumulative_episode_reward)
    if save_flag:
        plt.savefig("./{}/{}-{}-{}/cum_reward.png".format(env.spec.id, save_time.day, save_time.month, save_time.hour))
        plt.close()

    if save_flag:
        # Save hyperparameters in text file
        parameter_file = open("./{}/{}-{}-{}/parameters".format(env.spec.id, save_time.day, save_time.month,
                                                                save_time.hour), 'w')
        parameter_string = "Learning Epochs:{}\nMax Steps:{}\nActor:{}\nActor LR:{}\nCritic:{}\nCritic LR:{}" \
                           "\nGamma:{}\nTau:{}\nL2 Weight Decay:{}\nReplay Batch Size:{}\n Replay Total Size:{}\n" \
                           "OU Theta:{}\nOU Sigma:{}\nEpsilon:{}".\
            format(epochs, max_steps, actor_local.num_layers, lr_actor, critic_local.num_layers, lr_critic, gamma, tau,
                   weight_decay, memory.batch_size, memory.memory.maxlen, noise.theta, noise.sigma, epsilon)
        parameter_file.write(parameter_string)
        parameter_file.close()

    # Show plots
    if save_flag:
        # Save torch model of actor and critic
        # Saving in format day-month-hour-episode
        torch.save(actor_local.state_dict(), './{}/{}-{}-{}/actor'.
                   format(env.spec.id, save_time.day, save_time.month, save_time.hour))
        torch.save(critic_local.state_dict(), './{}/{}-{}-{}/critic'.
                   format(env.spec.id, save_time.day, save_time.month, save_time.hour))
    else:
        plt.show()

    # Return learned policy / actor network
    return actor_local


def main():
    """
    Defining the gym environment and initializing the DDPG objects (NNs, noise and replay buffer).
    Hyperparameters are set in this method.  # TODO: Define size of nn layers in this method
    Training and evaluation methods are executed.
    :return: None
    """

    global env
    env = gym.make('Qube-v0')
    # env = gym.make('CartpoleSwingShort-v0')
    # env = gym.make('Pendulum-v0')
    # env = gym.make('BallBalancerSim-v0')
    print(env.spec.id)
    print("State Space:\tShape:{}\tLow:{}\tHigh:{}".format(np.shape(env.reset()), env.observation_space.low,
                                                           env.observation_space.high))
    print("Action Space:\tShape:{}\tLow:{}\tHigh:{}".format(np.shape(env.action_space.sample()), env.action_space.low,
                                                            env.action_space.high))
    print("Reward Range:{}".format(env.reward_range))
    env_observation_size = len(env.reset())
    env_action_size = len(env.action_space.sample())
    env_action_low = env.action_space.low
    env_action_high = env.action_space.high
    global env_specs
    env_specs = (env_observation_size, env_action_size, env_action_low, env_action_high)
    random_seed = 3
    env.seed(3)

    # Noise generating process
    OU_NOISE = OUNoise(size=env_action_size, seed=random_seed, mu=0., theta=0.15, sigma=0.4)

    GAUSS_NOISE = Gaussian(size=env_action_size, seed=random_seed, mu=0.0, sigma=1.0, decay=0.0)

    # Replay memory
    MEMORY = ReplayBuffer(action_size=env_specs[1], buffer_size=int(1e6), batch_size=256,
                          seed=random_seed)

    # Run training procedure with defined hyperparameters
    ACTOR = training(epochs=10000, max_steps=10000, epoch_checkpoint=1000, noise=GAUSS_NOISE, epsilon=None,
                     epsilon_decrease=None, add_noise=True, lr_actor=1e-4, lr_critic=1e-3, weight_decay=0,
                     gamma=0.99, memory=MEMORY, tau=1e-3, seed=random_seed, save_flag=True, load_flag=False,
                     load_path='actor22-1-18', render=True)

    # Run evaluation
    # evaluation(load_flag=False, actor='./actor-21-2-16', epochs=25, render=False)
    evaluation(actor=ACTOR, epochs=25, render=False)


if __name__ == "__main__":
    main()
