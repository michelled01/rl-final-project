import multitask_atari
import gymnasium as gym

import numpy as np
import torch
from torch import optim
import warnings

from utils.logger import Logger
from utils.fixed_replay_buffer import WrappedFixedReplayBuffer, History
# from utils.preprocess import phi_map
from utils.dqn import DeepQNetwork, Q_targets, Q_values, copy_network, gradient_descent
# from learn.py import e_greedy_action

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
    

config = {
    'in_dir' : [
        "Pong/1/replay_logs",
        "Breakout/1/replay_logs"
    ],
    'log_dir' : "logs/",
    'log_freq' : 1,
    # seed for random, np.random
    #seed: 42,
    'atari-env-names': [
        'Pong',
        'Breakout',
    ],
    'algo': 'PPO',
    'policy': 'CnnPolicy',
    'total-timesteps': 216000, # 1 hour of gameplay
    'max-episode-steps': 600,  # 10 seconds of gameplay
}

def gen_seed(config):
    if 'seed' in config:
        import random, numpy as np, torch
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
    else:
        warnings.warn('random number generators have not been seeded')

def id(env_name):
    if "Pong" in env_name:
        return 0
    elif "Breakout" in env_name:
        return 1
    else:
        return -1


def main():
    gen_seed(config)

    env = gym.make('multitask-atari', max_episode_steps=config['max-episode-steps'], env_names=config['atari-env-names'], render_mode='rgb_array')
    env = gym.wrappers.HumanRendering(env)
    params = {
        'num_episodes': 40, # 4000
        'minibatch_size': 32,
        'max_episode_length': 200, # int(10e6),  # T
        'memory_size': int(4.5e5),  # N
        'history_size': 4,  # k
        'train_freq': 4,
        'target_update_freq': 10000,  # C: Target nerwork update frequency
        'num_actions': env.action_space.n,
        'min_steps_train': 50000,
        'cur_env_id': 0
    }

    step = 0
    
    log = Logger(log_dir=config['log_dir'])
    
    # Initialize replay memory D to capacity N
    D = [WrappedFixedReplayBuffer(data_dir=path, replay_suffix=0, observation_shape=(84, 84), stack_size=4) for path in config['in_dir']]
   
    # Initialize action-value function Q with random weights
    Q = DeepQNetwork(params['num_actions'])
    log.network(Q)
    
    Q_ = copy_network(Q)
    
    optimizer = optim.RMSprop(
        Q.parameters(), lr=0.00025, alpha=0.95, eps=.01
    )
    
    H = History.initial(env)

    for ep in range(params['num_episodes']):
        print(ep)

        phi = np.asarray(H.get())
        phi = np.transpose(phi,(3, 0, 1, 2))
        phi = torch.from_numpy(phi).float()

        for _ in range(params['max_episode_length']):
            step += 1
            # Select action a_t for current state s_t
            action, epsilon = e_greedy_action(Q, phi, env, step)
            if step % config['log_freq'] == 0:
                log.epsilon(epsilon, step)
            # Execute action a_t in emulator and observe reward r_t and obs x_(t+1)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Clip reward to range [-1, 1]
            reward = max(-1.0, min(reward, 1.0))
            if reward: print(reward)

            H.add(obs)
            new_phi = np.asarray(H.get())
            new_phi = np.transpose(new_phi,(3, 0, 1, 2))
            new_phi = torch.from_numpy(new_phi).float()
            phi_prev, phi = phi, new_phi
            
            D[params['cur_env_id']].add(phi_prev, action, reward, done)
            phi_mb, a_mb, r_mb, phi_plus1_mb, done_mb, indices = D[params['cur_env_id']].memory.sample_transition_batch(batch_size=params['minibatch_size'], indices=None)
            # Perform a gradient descent step on ( y_j - Q(phi_j, a_j) )^2
            y = Q_targets(phi_plus1_mb, r_mb, done_mb, Q_)
            q_values = Q_values(Q, phi_mb, a_mb)
            q_phi, loss = gradient_descent(y, q_values, optimizer)
            
            if step % (params['train_freq'] * config['log_freq']) == 0:
                log.q_loss(q_phi, loss, step)
            # Reset Q_
            if step % params['target_update_freq'] == 0:
                del Q_
                Q_ = copy_network(Q)

            log.episode(reward)

            # Restart game if done
            if done:
                H = History.initial(env)
                log.reset_episode()
                params['cur_env_id'] = id(str(env.get_cur_env()))
                print(params['cur_env_id'])
                break

if __name__ == '__main__':
    main()