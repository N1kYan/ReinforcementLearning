from DiscreteEnvironment import DiscreteEnvironment
import gym
import quanser_robots

env = gym.make('Pendulum-v2')
disc_env = DiscreteEnvironment(env=env, name='EasyPendulum', state_space_size=(4+1, 4+1), action_space_size=(4+1,))


disc_env.update_transition_probabilities(policy=None, epochs=25000)
