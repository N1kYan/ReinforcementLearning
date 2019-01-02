import tensorflow as tf
import tflearn
import numpy as np
import gym
import quanser_robots
import argparse
import pprint as pp
import random
from gym import wrappers
from collections import deque


class ReplayBuffer(object):
    """
        Class for the replay buffer
        - storing experience samples
        - sampling random batches from storage

    """
    def __init__(self, buffer_size, random_seed=123):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, state, action, reward, t, next_state):
        experience = (state, action, reward, t, next_state)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count +=1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        state_batch = np.array([_[0] for _ in batch])
        action_batch = np.array([_[1] for _ in batch])
        reward_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        next_state_batch = np.array([_[4] for _ in batch])

        return state_batch, action_batch, reward_batch, t_batch, next_state_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0


class ActorNetwork(object):
    """
        Class for the actor network

        The input is the current state of the agent,
        the output is the action under a deterministic policy.

        (tanh in the output layer keeps the action between [-1, 1] and then gets
        rescaled to the action bounds)

    """
    def __init__(self, tf_session, state_dim, action_dim, action_bound, lr, tau, batch_size):
        self.sess = tf_session
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = lr
        self.tau = tau
        self.batch_size = batch_size

        # Actor network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()
        self.network_parameters = tf.trainable_variables()

        # Target actor network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()
        self.target_parameters = tf.trainable_variables()[len(self.network_parameters):]

        # Operation to update the target network parameters (with online network weights)
        # See DDPG paper
        self.update_target_parameters = \
            [self.target_parameters[i].assign(tf.multiply(self.network_parameters[i], self.tau) +
                                              tf.multiply(self.target_parameters[i], 1.0 - self.tau))
             for i in range(len(self.target_parameters))]

        # Gradient placeholder for the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.action_dim])

        # Combining the gradients ???
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_parameters, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Operation for ADAM optimization
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_parameters))

        self.num_trainable_vars = len(self.network_parameters) + len(self.target_parameters)

    def create_actor_network(self):
        # Input layer
        inputs = tflearn.input_data(shape=[None, self.state_dim])
        # First hidden layer
        h1 = tflearn.fully_connected(inputs, 400)  # fc layer with 400 neurons
        h1 = tflearn.layers.normalization.batch_normalization(h1)
        h1 = tflearn.activations.relu(h1)
        # Second hidden layer
        h2 = tflearn.fully_connected(h1, 300)  # fc layer with 300 neurons
        h2 = tflearn.layers.normalization.batch_normalization(h2)
        h2 = tflearn.activations.relu(h2)
        # Initialize weights of output layer
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)  # Why do we init only w of last layer?
        # Output layer
        out = tflearn.fully_connected(h2, self.action_dim, activation='tanh', weights_init=w_init)
        scaled_out = tf.multiply(out, self.action_bound)  # Why do we need this? Because of tanh?
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_parameters)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
        Class for the critic network
        - Skips the first hidden layer for the action inputs

        Input to the network is the state and action, output is Q(s,a).
        The action must be obtained from the output of the actor network.

    """
    def __init__(self, tf_session, state_dim, action_dim, lr, tau, gamma, num_actor_vars):
        self.sess = tf_session
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = lr
        self.tau = tau
        self.gamma = gamma

        # Create critic network
        self.inputs, self.action, self.out = self.create_critic_network()
        self.network_parameters = tf.trainable_variables()

        # Create critic target
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()
        self.target_parameters = tf.trainable_variables()[(len(self.network_parameters) + num_actor_vars):]

        # Operation to update the target network parameters (with online network weights)
        # See DDPG paper
        self.update_target_parameters = \
            [self.target_parameters[i].assign(tf.multiply(self.network_parameters[i], self.tau) +
                                              tf.multiply(self.target_parameters[i], 1.0 - self.tau))
             for i in range(len(self.target_parameters))]

        # Network target y_i obtained from the target networks
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Operations for loss and optimization
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Gradients of the network w.r.t. the actions
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        # State input layer
        state_inputs = tflearn.input_data(shape=[None, self.state_dim])
        # Action input layer
        action_inputs = tflearn.input_data(shape=[None, self.action_dim])
        # First hidden layer
        h1 = tflearn.fully_connected(state_inputs, 400)
        h1 = tflearn.layers.normalization.batch_normalization(h1)
        h1 = tflearn.activations.relu(h1)
        # Two more hidden layers, now also including the action(s)
        h2_state = tflearn.fully_connected(h1, 300)
        h2_action = tflearn.fully_connected(action_inputs, 300)
        # Merge layers
        h3 = tflearn.activation(tf.matmul(h1, h2_state.W) + tf.matmul(action_inputs, h2_action.W) + h2_action.b,
                                activation='relu')
        # Initialize weights of output layer
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        # Output layer
        out = tflearn.fully_connected(h3, 1, weights_init=w_init)
        return state_inputs, action_inputs, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_parameters)


# TODO: Add source?
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteilUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


def build_summaries():
    """
        Tensorflow summary ops
    :return:
    """
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)
    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()
    return summary_ops, summary_vars


def train(sess, env, args, actor, critic, actor_noise):
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    # Needed to enable BatchNorm.
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    # tflearn.is_training(True)

    for i in range(int(args['max_episodes'])):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(int(args['max_episode_len'])):

            if args['render_env']:
                env.render()

            # Added exploration noise
            # a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()

            s2, r, terminal, info = env.step(a[0])

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            if terminal:
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j)
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), \
                                                                             i, (ep_ave_max_q / float(j))))
                break


def main(args):
    with tf.Session() as sess:

        env = gym.make(args['env'])
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        assert (env.action_space.high == -env.action_space.low)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())

        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        if args['use_gym_monitor']:
            if not args['render_env']:
                env = wrappers.Monitor(
                    env, args['monitor_dir'], video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, args['monitor_dir'], force=True)

        train(sess, env, args, actor, critic, actor_noise)

        if args['use_gym_monitor']:
            env.monitor.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_ddpg')

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=True)

    args = vars(parser.parse_args())

    pp.pprint(args)

main(args)






"""
def main():
    TAU = 0.001
    ACTOR_LEARNING_RATE = 0.0001
    CRITIC_LEARNING_RATE = 0.001
    BUFFER_SIZE = 1000000
    MINIBATCH_SIZE = 64
    MAX_EPISODES = 50000
    MAX_EP_STEPS = 1000
    GAMMA = 0.99
    RENDER_ENV = False

    with tf.Session() as sess:
        env = gym.make('Pendulum-v0')
        env_state_dim = env.observation_space.shape[0]
        env_action_dim = env.action_space.shape[0]
        env_action_bound = env.action_space.high  # ??
        assert (env.action_space.high == -env.action_space.low)  # ??
        # Create actor and critic networks
        actor = ActorNetwork(sess, env_state_dim, env_action_dim, env_action_bound,
                             ACTOR_LEARNING_RATE, TAU, MINIBATCH_SIZE)
        critic = CriticNetwork(sess, env_state_dim, env_action_dim, env_action_bound,
                               CRITIC_LEARNING_RATE, TAU,
                               actor.get_num_trainable_vars())
        # Noise for the actor
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(env_action_dim))
        # Initialize tf variables
        sess.run(tf.initialize_all_variables())
        # Initialize target network weights
        actor.update_target_network()
        critic.update_target_network()
        # Initialize replay buffer
        replay_buffer = ReplayBuffer(BUFFER_SIZE)
        # Run!
        for e in range(MAX_EPISODES):
            # Draw initial state
            state = env.reset()
            if RENDER_ENV or e >= MAX_EPISODES-5:
                env.render()
            # Predict action with actor network and add noise
            action = actor.predict(np.reshape(state, (1, actor.state_dim))) + actor_noise()
            # Observe next state, reward, ...
            next_state, reward, done, info = env.step(action[0])
            # Add observation to replay buffer
            replay_buffer.add(np.reshape(state, (actor.state_dim,)),
                              np.reshape(action, (actor.action_dim,)),
                              reward,
                              done,
                              np.reshape(next_state, (actor.state_dim,)))

            # Only continue if replay buffer has at least the size of one minibatch
            if replay_buffer.size() > MINIBATCH_SIZE:
                state_batch, action_batch, reward_batch, t_batch, next_state_batch = \
                    replay_buffer.sample_batch(MINIBATCH_SIZE)

                # Calculate targets
                target_q = critic.predict_target(next_state_batch, actor.predict_target(next_state_batch))

                y_i = []
                for k in range(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(reward_batch[k])
                    else:
                        y_i.append(reward_batch[k] + GAMMA * target_q[k])

                # Update critic given the targets
                predicted_q_value, _ = critic.train(state_batch, action_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                # Update the actor policy using the sampled gradient
                action_outputs = actor.predict(state_batch)
                gradients = critic.action_gradients(state_batch, action_outputs)
                actor.train(state_batch, gradients[0])

                # Update the target networks
                actor.update_target_network()
                critic.update_target_network()

            if done:
                break


if __name__ == main():
    main()
"""

