import tensorflow as tf

import gym
import quanser_robots


def policy_gradient(env, learning_rate):

    with tf.variable_scope("policy"):

        # ------------------ Read Dimensions ---------------------- #

        # TODO: Actions can have several dimensions
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        act_state_dim = state_dim * action_dim

        # -------------------- Input Variables -------------------- #
        pl_state_input = tf.placeholder("float",
                                        [None, state_dim],
                                        name="pl_state_input")

        pl_actions_input = tf.placeholder("float",
                                          [None, action_dim],
                                          name="pl_actions_input")

        pl_advantages_input = tf.placeholder("float", [None, ],
                                             name="pl_advantages_input")

        # ------------------------ Weights ------------------------ #

        pl_weights = tf.get_variable("pl_weights", [state_dim, action_dim])

        # ------------------------ Network ------------------------ #

        # This is our network. It is simple, linear
        # and has just 1 weight tensor.
        linear = tf.matmul(pl_state_input, pl_weights)

        # Softmax function: sum(probabilities) = 1
        pl_probabilities = tf.nn.softmax(linear)

        # ------------------- Trainable Vars ------------------------ #

        # Returns a list which only contains pl_weights, as it is our only
        # trainable variable: [tf.Variable with shape =(4, 2)]
        pl_train_vars = tf.trainable_variables()

        # ------------------------ π(a|s) -------------------------- #

        action_prob = tf.multiply(pl_probabilities, pl_actions_input)

        action_prob = tf.reduce_sum(action_prob, axis=[1])

        # ----------------------- log(π(a|s)) ----------------------- #

        action_log_prob = tf.log(action_prob)

        # ------------------- ∇_θ log(π(a|s)) ----------------------- #
        # Calculate the gradient of the log probability at each point in time

        # NOTE: doing this because tf.gradients only returns a summed version
        # TODO: As far as I can tell, this is unnecessary. The array is already
        #   flattened. Maybe when we have high dim actions it will be useful.
        # Results in shape (200,)
        action_log_prob_flat = tf.reshape(action_log_prob, (-1,))

        # TODO: Do we want to take gradient of scalar and all weights
        # Take the gradient of each action w.r.t. the trainable weights
        # Results in shape (200, 4, 2): List with 200 tensors of shape (4, 2)
        g_log_prob = [tf.gradients(action_log_prob_flat[i], pl_train_vars)[0]
                      for i in range(action_log_prob_flat.get_shape()[0])]

        # Results in shape (200, 4, 2)
        g_log_prob = tf.stack(g_log_prob)

        # Results in shape (200, 8, 1)
        g_log_prob = tf.reshape(g_log_prob, (g_log_prob.shape[0], act_state_dim, 1))

        # ------------------- ∇_θ J(θ) ----------------------- #

        # Restuls in shape (200, 1, 1)
        adv_reshaped = tf.reshape(pl_advantages_input, (g_log_prob.shape[0], 1, 1))

        # Results in shape (200, 8, 1)
        grad_j = tf.multiply(g_log_prob, adv_reshaped)

        # Get the mean (sum over time and divide by 1/time steps) to get the
        # expectation E. Results in shape (8, 1).
        grad_j_steps = grad_j.shape[0]
        grad_j = tf.reduce_sum(grad_j, reduction_indices=[0])
        grad_j = 1.00 / grad_j_steps * grad_j

        # --------------------- δθ = Policy Update --------------------- #
        # We calculate the natural gradient policy update:
        # δθ = α x ∇_θ J(θ)

        # Multiply natural gradient by a learning rate
        pl_update = tf.multiply(learning_rate, grad_j)

        # Reshape to (2, 4) because our weight tensor has this shape
        pl_update = tf.reshape(pl_update, (state_dim, action_dim))

        # Update trainable parameters which in our case is just one tensor
        # NOTE: Whenever pl_train_vars is fetched they're also updated
        pl_train_vars[0] = tf.assign_add(pl_train_vars[0], pl_update)

        return pl_state_input, pl_actions_input, pl_advantages_input, \
            pl_probabilities, pl_train_vars
