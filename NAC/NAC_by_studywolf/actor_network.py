import tensorflow as tf

import gym
# import quanser_robots

def policy_gradient(env):
    """
    Neural Network to approximate our policy.

    :param env: the environment we are optimizing
    :return:
    """
    with tf.variable_scope("policy"):

        # TODO: Actions can have several dimensions
        state_dim = env.observation_space.shape[0]  # dim of one state
        action_dim = len(env.action_space)

        # -------------------- Input Variables -------------------- #

        # During runtime we will feed the state(-vector) to the network
        # using this variable.
        pl_state_input = tf.placeholder("float",
                                        [None, state_dim],
                                        name="pl_state_input")

        # - We have to specify shape of actions so we can call get_shape
        # when calculating g_log_prob below. -
        # During runtime we will use this variable to input all the actions
        # which appeared during our episode into our policy network.
        # The amount of action during an episode is a fixed value which has
        # been predefined by the user. Each action is displayed by a one-hot
        # array. All entries are 0, except the action that was taken, which has
        # a one. This action was chosen stochastically regarding the
        # probabilites of the policy network.
        pl_actions_input = tf.placeholder("float",
                                          [env.time_steps, action_dim],
                                          name="pl_actions_input")

        # Placeholder with just 1 dimension which is dynamic
        # We use it to feed the advantages of our episode to the network. The
        # size of the tensor is determined by the number of steps the agent
        # executes during one episode run.
        pl_advantages_input = tf.placeholder("float", [None, ],
                                             name="pl_advantages_input")

        # ------------------------ Weights ------------------------ #

        pl_weights = tf.get_variable("pl_weights", [state_dim, action_dim])

        # ------------------------ Network ------------------------ #

        # This is our network.
        # It is simple, linear and just has 1 weight tensor.
        linear = tf.matmul(pl_state_input, pl_weights)

        # Softmax function: sum(probabilities) = 1
        probabilities = tf.nn.softmax(linear)
        print("Prob shape:", probabilities.shape)
        my_variables = tf.trainable_variables()

        # calculate the probability of the chosen action given the state
        action_log_prob = tf.log(tf.reduce_sum(
            tf.multiply(probabilities, pl_actions_input),
            axis=[1]
        ))

        # calculate the gradient of the log probability at each point in time
        # NOTE: doing this because tf.gradients only returns a summed version
        action_log_prob_flat = tf.reshape(action_log_prob, (-1,))

        g_log_prob = tf.stack(
            [tf.gradients(action_log_prob_flat[i], my_variables)[0]
                for i in range(action_log_prob_flat.get_shape()[0])])
        g_log_prob = tf.reshape(g_log_prob, (env.time_steps, action_dim * state_dim, 1))

        # calculate the policy gradient by multiplying by the advantage fct.
        g = tf.multiply(g_log_prob, tf.reshape(pl_advantages_input, (env.time_steps, 1, 1)))
        # sum over time
        g = 1.00 / env.time_steps * tf.reduce_sum(g, reduction_indices=[0])

        # --------------- Fischer Information Matrix --------------- #

        # calculate the Fischer information matrix and its inverse
        F2 = tf.map_fn(lambda x: tf.matmul(x, tf.transpose(x)), g_log_prob)
        F = 1.0 / env.time_steps * tf.reduce_sum(F2, reduction_indices=[0])

        # ------------------------ SVD Clip ------------------------ #

        # calculate inverse of positive definite clipped F
        # NOTE: have noticed small eigenvalues (1e-10) that are negative,
        # using SVD to clip those out, assuming they're rounding errors
        S, U, V = tf.svd(F)
        atol = tf.reduce_max(S) * 1e-6
        S_inv = tf.divide(1.0, S)
        S_inv = tf.where(S < atol, tf.zeros_like(S), S_inv)
        S_inv = tf.diag(S_inv)
        F_inv = tf.matmul(S_inv, tf.transpose(U))
        F_inv = tf.matmul(V, F_inv)

        # --------------------- (Policy?) Update --------------------- #

        # calculate natural policy gradient ascent update
        F_inv_g = tf.matmul(F_inv, g)
        # calculate a learning rate normalized such that a constant change
        # in the output control policy is achieved each update, preventing
        # any parameter changes that hugely change the output
        learning_rate = tf.sqrt(
            tf.divide(0.001, tf.matmul(tf.transpose(g), F_inv_g)))

        update = tf.multiply(learning_rate, F_inv_g)
        update = tf.reshape(update, (state_dim, action_dim))

        # update trainable parameters
        # NOTE: whenever my_variables is fetched they're also updated
        my_variables[0] = tf.assign_add(my_variables[0], update)

        return pl_state_input, pl_actions_input, pl_advantages_input, \
               probabilities, my_variables
