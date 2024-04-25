import logging
logging.getLogger().setLevel(logging.INFO)
import gymnasium as gym
import numpy as np
from torch import optim

from utils.logger import Logger
from utils.fixed_replay_buffer import WrappedFixedReplayBuffer, History
from utils.preprocess import phi_map
from utils.dqn import DeepQNetwork, Q_targets, Q_values, save_network, copy_network, gradient_descent

flags = {
    'out_dir' : "Pong/1",
    'in_dir' : "Pong/1/replay_logs",
    'log_dir' : "logs/",
    'log_freq' : 1,
    'save_freq' : 100
}

# Learning

def e_greedy_action(Q, phi, env, step):
    print("phi is",phi.shape, phi.size)
    # Initial values
    initial_epsilon, final_epsilon = 1.0, .1
    # Decay steps
    decay_steps = float(1e6)
    # Calculate step size to move from final to initial epsilon with #decay_steps
    step_size = (initial_epsilon - final_epsilon) / decay_steps
    # Calculate annealed epsilon
    ann_eps = initial_epsilon - step * step_size
    # Define allowsd min. epsilon
    min_eps = 0.1
    # Set epsilon as max(min_eps, annealed_epsilon)
    epsilon = max(min_eps, ann_eps)

    # Obtain a random value in range [0,1)
    rand = np.random.uniform()

    # With probability e select random action a_t
    if 0 and rand < epsilon:
        return env.action_space.sample(), epsilon

    else:
        # Otherwise select action that maximises Q(phi)
        # In other words: a_t = argmax_a Q(phi, a)
        import torch
        phi = torch.from_numpy(phi).float()
        max_q = Q(phi).max(1)[1]
        return max_q.data[0], epsilon
    

# Tranining

env = gym.make('ALE/Pong-v5', render_mode=None)
# Current iteration
step = 0
# Has trained model
has_trained_model = False
# Init training params
params = {
    'num_episodes': 40, # 4000
    'minibatch_size': 32,
    'max_episode_length': 200, # int(10e6),  # T
    'memory_size': int(4.5e5),  # N
    'history_size': 4,  # k
    'train_freq': 4,
    'target_update_freq': 10000,  # C: Target nerwork update frequency
    'num_actions': env.action_space.n,
    'min_steps_train': 50000
}
# Initialize logger
log = Logger(log_dir=flags['log_dir'])
# Initialize replay memory D to capacity N
D = WrappedFixedReplayBuffer(data_dir=flags['in_dir'], replay_suffix=0, observation_shape=(84, 84), stack_size=4)
skip_fill_memory = True
# Initialize action-value function Q with random weights
Q = DeepQNetwork(params['num_actions'])
log.network(Q)
# Initialize target action-value function Q^
Q_ = copy_network(Q)
# Init network optimizer
optimizer = optim.RMSprop(
    Q.parameters(), lr=0.00025, alpha=0.95, eps=.01  # ,momentum=0.95,
)
# Initialize sequence s1 = {x1} and preprocessed sequence phi = phi(s1)
H = History.initial(env)

for ep in range(params['num_episodes']):
    print(ep)

    phi = phi_map(H.get())
    # del phi

    if (ep % flags['save_freq']) == 0:
        save_network(Q, ep, out_dir=flags['out_dir'])

    for _ in range(params['max_episode_length']):
        '''not on windows'''
        # if step % 100 == 0:
        #     print ('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        step += 1
        # Select action a_t for current state s_t
        action, epsilon = e_greedy_action(Q, phi, env, step)
        if step % flags['log_freq'] == 0:
            log.epsilon(epsilon, step)
        # Execute action a_t in emulator and observe reward r_t and image x_(t+1)
        image, reward, done, _, _ = env.step(action)

        # Clip reward to range [-1, 1]
        reward = max(-1.0, min(reward, 1.0))
        # Set s_(t+1) = s_t, a_t, x_(t+1) and preprocess phi_(t+1) =  phi_map( s_(t+1) )
        H.add(image)
        phi_prev, phi = phi, phi_map(H.get())
        # Store transition (phi_t, a_t, r_t, phi_(t+1)) in D
        D.add(phi_prev, action, reward, done)

        should_train_model = skip_fill_memory or \
            ((step > params['min_steps_train']) and
             D.can_sample(params['minibatch_size']) and
             (step % params['train_freq'] == 0))

        if should_train_model:
            if not (skip_fill_memory or has_trained_model):
                D.save(params['min_steps_train'])
            has_trained_model = True

            # Sample random minibatch of transitions ( phi_j, a_j, r_j, phi_(j+1)) from D
            phi_mb, a_mb, r_mb, phi_plus1_mb, done_mb, indices = D.memory.sample_transition_batch(batch_size=params['minibatch_size'], indices=None)
            # Perform a gradient descent step on ( y_j - Q(phi_j, a_j) )^2
            y = Q_targets(phi_plus1_mb, r_mb, done_mb, Q_)
            q_values = Q_values(Q, phi_mb, a_mb)
            q_phi, loss = gradient_descent(y, q_values, optimizer)
            # Log Loss
            if step % (params['train_freq'] * flags['log_freq']) == 0:
                log.q_loss(q_phi, loss, step)
            # Reset Q_
            if step % params['target_update_freq'] == 0:
                del Q_
                Q_ = copy_network(Q)

        log.episode(reward)
        # if FLAGS["log_console"]:
        #     log.display()

        # Restart game if done
        if done:
            H = History.initial(env)
            log.reset_episode()
            break
