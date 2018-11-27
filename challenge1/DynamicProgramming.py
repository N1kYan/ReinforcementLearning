import numpy as np

"""
   Value Iteration

   Inputs:
   disc: Discriminator object
   theta: Minimal value function difference for convergence
   gamma: Update learning rate

"""


# TODO: Good choice of a learning rate (gamma)

def value_iteration(disc, theta, gamma):
    print("Starting Value iteration:")

    value_function = np.zeros(shape=disc.state_space_size)
    policy = np.zeros(shape=disc.state_space_size)

    delta = theta

    while_loop_num = 0

    while delta >= theta:

        delta = 0

        print("Loop number {} ... ".format(while_loop_num), end='')
        while_loop_num += 1

        # Iterate over discrete state space
        for j, s0 in enumerate(disc.state_space[0]):  # degrees
            for s1 in disc.state_space[1]:  # angular velocity

                # Get (only positive) indexes for (possibly negative) discrete state(s)
                index = disc.map_to_index([s0, s1])
                # print(index)

                v = value_function[index[0], index[1]]

                # Iterate over all actions to get action maximizing expected reward
                amax = 2
                rmax = -100

                for a in disc.action_space:
                    # Get sufficient state and reward from regressors
                    x = np.array([s0, s1, a])
                    x = x.reshape(1, -1)
                    next_s = regressorState.predict(x).T.reshape(-1, )
                    r = regressorReward.predict(x)

                    # Discretize sufficient state
                    next_index = disc.map_to_index([next_s[0], next_s[1]])

                    # Calculate expected reward
                    # Deterministic case; we do not need probability distribution
                    expected_reward = r + gamma * value_function[next_index[0], next_index[1]]

                    if rmax < expected_reward:
                        amax = a
                        rmax = expected_reward

                        # Define value function by maximum expected reward per state
                value_function[index[0], index[1]] = rmax
                # Define policy by action achieving maximum expected reward per state
                policy[index[0], index[1]] = amax
                # Update delta
                delta = max(delta, np.abs(v - value_function[index[0], index[1]]))

            # if s0 == int((disc.state_space_size[0]-1)*0.25):
            #    print("25%...")
            # if s0 == int((disc.state_space_size[0]-1)*0.5):
            #   print("50%...")
            # if s0 == int((disc.state_space_size[0]-1)*0.75):
            #    print("75%...")

        print("Done! (Delta = {})".format(delta))

    print()
    print("... done!")
    return value_function, policy


"""
    Policy Iteration

    Gives convergence towards the optimal policy by iteratively
    performing Policy Evaluation and Policy Improvement

"""


def policy_iteration(disc, theta, gamma):
    print("Policy iteration...")

    value_function = np.ones(shape=disc.state_space_size)
    policy = np.zeros(shape=disc.state_space_size)

    def policy_evaluation(theta, gamma):
        print()
        print("Evaluating policy")
        delta = theta
        while delta >= theta:
            delta = 0
            # Iteratate over discrete state space
            for s0 in disc.state_space[0]:
                for s1 in disc.state_space[1]:
                    for s2 in disc.state_space[2]:
                        # Get index for state
                        # The method already iterates over a discretized state space
                        # But the states need to get mapped to a positive index do to possible 'negative' states
                        index = disc.map_to_index([s0, s1, s2])

                        v = value_function[index[0], index[1], index[2]]

                        """
                         V(s) = Sum...p(s',r|s,pi(s))[r+gamma*V(s')]

                        """
                        a = policy[index[0], index[1], index[2]]

                        # input for regression
                        x = np.array([s0, s1, s2, a]).reshape(1, -1)

                        # Predict next state and reward with regressors
                        next_s = regressorState.predict(x).T.reshape(-1, )
                        r = regressorReward.predict(x)

                        next_index = disc.map_to_index([next_s[0], next_s[1], next_s[2]])

                        value_function[index[0], index[1], index[2]] = r + gamma * value_function[next_index[0],
                                                                                                  next_index[1],
                                                                                                  next_index[2]]

                        delta = max(delta, v - value_function[index[0], index[1], index[2]])
            print("Delta: ", delta)

    def policy_improvement(gamma):
        print()
        print("Improving policy")
        policy_stable = True
        for s0 in disc.state_space[0]:
            for s1 in disc.state_space[1]:
                for s2 in disc.state_space[2]:

                    # Indexing
                    index = disc.map_to_index([s0, s1, s2])

                    old_action = policy[index[0], index[1], index[2]]

                    """
                        pi(s) = argmax_a ... 
                        We do not have to care about the prob. distribution,
                        as we have a deterministic env.
    
                    """
                    # Iterate over all actions and get the one with max. expected reward
                    amax = 2
                    rmax = -100
                    for a in disc.action_space:
                        x = np.array([s0, s1, s2, a])
                        x = x.reshape(1, -1)
                        next_s = regressorState.predict(x).T.reshape(-1, )
                        next_index = disc.map_to_index([next_s[0], next_s[1], next_s[2]])
                        r = regressorReward.predict(x)
                        expected_reward = r + gamma * value_function[next_index[0], next_index[1], next_index[2]]
                        if rmax < expected_reward:
                            amax = a
                            rmax = expected_reward
                    policy[index[0], index[1], index[2]] = amax  # TODO

                    if old_action != policy[index[0], index[1], index[2]]:
                        policy_stable = False

        print("Policy stable: ", policy_stable)
        return policy_stable

    # Run until policy is stable
    stable_policy = False
    while not stable_policy:
        policy_evaluation(theta, gamma)
        stable_policy = policy_improvement(gamma)

    print()
    print("...done")
    return value_function, policy