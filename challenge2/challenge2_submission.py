"""
Submission template for Programming Challenge 2: Approximate Value Based Methods.


Fill in submission info and implement 4 functions:

- load_dqn_policy
- train_dqn_policy
- load_lspi_policy
- train_lspi_policy

Keep Monitor files generated by Gym while learning within your submission.
Example project structure:

challenge2_submission/
  - challenge2_submission.py
  - dqn.py
  - lspi.py
  - dqn_eval/
  - dqn_train/
  - lspi_eval/
  - lspi_train/
  - supplementary/

Directories `dqn_eval/`, `dqn_train/`, etc. are autogenerated by Gym (see below).
Put all additional results into the `supplementary` directory.

Performance of the policies returned by `load_xxx_policy` functions
will be evaluated and used to determine the winner of the challenge.
Learning progress and learning algorithms will be checked to confirm
correctness and fairness of implementation. Supplementary material
will be manually analyzed to identify outstanding submissions.
"""

import numpy as np
import torch
import DQN
import pickle
import LSPI


info = dict(
    group_number=None,  # change if you are an existing seminar/project group
    authors="John Doe; Lorem Ipsum; Foo Bar",
    description="Explain what your code does and how. "
                "Keep this description short "
                "as it is not meant to be a replacement for docstrings "
                "but rather a quick summary to help the grader.")


def load_dqn_policy():
    """
    Load pretrained DQN policy from file.

    The policy must return a continuous action `a`
    that can be directly passed to `CartpoleSwingShort-v0` env.

    :return: function pi: s -> a
    """
    model = torch.load("model.pt")
    model.eval()
    return model
    #return lambda obs: np.array([3.1415])


def train_dqn_policy(env):
    """
    Execute your implementation of the DQN learning algorithm.

    This function should start your code placed in a separate file.

    :param env: gym.Env
    :return: function pi: s -> a
    """
    model = DQN.run_dqn(env)
    return model
    #return lambda obs: np.array([2.7183])


def load_lspi_policy():
    """
    Load pretrained LSPI policy from file.

    The policy must return a continuous action `a`
    that can be directly passed to `CartpoleStabShort-v0` env.

    :return: function pi: s -> a
    """
    w_star = pickle.load(open("lspi_weights.p", "rb"))
    return lambda obs: LSPI.pi(obs, w_star)


def train_lspi_policy(env):
    """
    Execute your implementation of the LSPI learning algorithm.

    This function should start your code placed in a separate file.

    :param env: gym.Env
    :return: function pi: s -> a
    """
    w_star = LSPI.train(my_env=env)
    return lambda obs: LSPI.pi(obs, w_star)


# ==== Example evaluation
def main():
    import gym
    from gym.wrappers.monitor import Monitor
    import quanser_robots

    def evaluate(env, policy, num_evlas=25):
        ep_returns = []
        for eval_num in range(num_evlas):
            episode_return = 0
            dones = False
            obs = env.reset()
            while not dones:
                action = policy(obs)
                obs, rewards, dones, info = env.step(action)
                episode_return += rewards
            ep_returns.append(episode_return)
        return ep_returns

    def render(env, policy):
        obs = env.reset()
        done = False
        while not done:
            env.render()
            act = policy(obs)
            obs, _, done, _ = env.step(act)

    def check(env, policy):
        render(env, policy)
        ret_all = evaluate(env, policy)
        print(np.mean(ret_all), np.std(ret_all))
        env.close()

    # DQN I: Check learned policy
    env = Monitor(gym.make('CartpoleSwingShort-v0'), 'dqn_eval', force=True)
    policy = load_dqn_policy()
    check(env, policy)

    # DQN II: Check learning procedure
    env = Monitor(gym.make('CartpoleSwingShort-v0'), 'dqn_train', video_callable=False, force=True)
    policy = train_dqn_policy(env)
    check(env, policy)
    '''
    # LSPI I: Check learned policy
    env = Monitor(gym.make('CartpoleStabShort-v0'), 'lspi_eval')
    policy = load_lspi_policy()
    check(env, policy)

    # LSPI II: Check learning procedure
    env = Monitor(gym.make('CartpoleStabShort-v0'), 'lspi_train', video_callable=False)
    policy = train_lspi_policy(env)
    check(env, policy)'''


if __name__ == '__main__':
    main()
