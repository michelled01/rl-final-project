def e_greedy_action(Q, phi, env, step):
    initial_epsilon, final_epsilon = 1.0, .1
    decay_steps = float(1e6)
    step_size = (initial_epsilon - final_epsilon) / decay_steps
    ann_eps = initial_epsilon - step * step_size
    min_eps = 0.1
    epsilon = max(min_eps, ann_eps)

    rand = np.random.uniform()

    if rand < epsilon:
        return env.action_space.sample(), epsilon
    else:
        # a_t = argmax_a Q(phi, a)
        max_q = Q(phi).max(1)[1]
        return max_q.data[0], epsilon
    