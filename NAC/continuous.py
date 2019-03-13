if self.env.continuous:
    # Not implemented yet
    obs_vector = np.expand_dims(observation, axis=0)
    (act_state_input, _, _, act_probabilities, _) \
        = self.actor.get_net_variables()
    action = sess.run(
        act_probabilities,
        feed_dict={act_state_input: obs_vector})
    batch_actions.append(action)

else: