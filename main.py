import multitask_atari
import gymnasium as gym
import numpy as np
import torch
import warnings
from PIL import Image

from utils.logger import Logger
from utils.fixed_replay_buffer import WrappedFixedReplayBuffer, History
from utils.dqn import DeepQNetwork, Q_targets, Q_values, copy_network, save_network, gradient_descent
from utils.learn import e_greedy_action

config = {
    'in_dir' : [
        "Pong/1/replay_logs",
        "Breakout/1/replay_logs"
    ],
    'out_dir' : "Pong_and_Breakout",
    'action_mappings' : [
        np.array([0,1,3,4,11,12,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1], dtype=np.int32),
        np.array([0,1,3,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1], dtype=np.int32),
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
    'frame_stack_size': 4,
    'save_model_freq': 10
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
    env = gym.wrappers.FrameStack(env, config['frame_stack_size'])
    env = gym.wrappers.HumanRendering(env)
    params = {
        'num_episodes': 40,
        'minibatch_size': 4,
        'max_episode_length': 100, #int(10e6),  # T
        'memory_size': int(4.5e5),  # N
        'history_size': 4,  # k
        'train_freq': 1,
        'target_update_freq': 10000,  # C: Target nerwork update frequency
        'num_actions': env.action_space.n,
        'min_steps_train': 50000,
        'cur_env_id': 0
    }

    step = 0
    
    log = Logger(log_dir=config['log_dir'])
    
    # Initialize replay memory D to capacity N
    D = [WrappedFixedReplayBuffer(data_dir=path, replay_suffix=0, action_mappings=config['action_mappings'][i], observation_shape=(84, 84), stack_size=config['frame_stack_size']) for i,path in enumerate(config['in_dir'])]
   
    # Initialize action-value function Q with random weights
    Q = DeepQNetwork(params['num_actions'])
    log.network(Q)
    
    Q_ = copy_network(Q)
    
    optimizer = torch.optim.RMSprop(
        Q.parameters(), lr=0.0009722687647902346, alpha=0.9999, eps=.04629326552135021
    )
    
    H = History.initial(env)

    for ep in range(params['num_episodes']):
        print(ep)

        phi = np.asarray(H.get())
        # ck similarity to the other transpose using PIL to images
        # img = Image.fromarray(np.transpose(phi[0, 0, :, :])) # first value framestacking
        # img.save(f"img{step}.png")
        phi = torch.from_numpy(phi).float()

        if (ep % config['save_model_freq']) == 0:
            save_network(Q, ep, out_dir=config['out_dir'])

        for _ in range(params['max_episode_length']):
            step += 1
            # Select action a_t for current state s_t
            action, epsilon = e_greedy_action(Q, phi, env, step)

            if step % config['log_freq'] == 0:
                log.epsilon(epsilon, step)
                log.reset_episode()
            # Execute action a_t in emulator and observe reward r_t and obs x_(t+1)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if reward: print(reward)

            H.add(obs)
            new_phi = np.asarray(H.get())
            # img = Image.fromarray(np.transpose(new_phi[0, 0, :, :])) # first value framestacking
            # img.save(f"img2{step}.png")
            new_phi = torch.from_numpy(new_phi).float()
            phi_prev, phi = phi, new_phi

            D[params['cur_env_id']].add(phi_prev, action, reward, done)
            phi_mb, a_mb, r_mb, phi_plus1_mb, done_mb, indices = D[params['cur_env_id']].memory.sample_transition_batch(batch_size=params['minibatch_size'], indices=None)
            # a_mb = (config['action_mappings'][params['cur_env_id']])[np.array(a_mb)]

            phi_mb = np.transpose(phi_mb,(1,0,2,3))
            phi_plus1_mb = np.transpose(phi_plus1_mb,(1,0,2,3))
            # these images should be relatively similar
            # phi1 = phi_mb
            # phi2 = phi_prev.cpu().detach().numpy().astype(np.uint8)
            # img = Image.fromarray(np.transpose(phi1[0, :3, :, :]))
            # img.save(f"phi{step}.png")
            # img = Image.fromarray(np.transpose(phi2[0, :3, :, :]))
            # img.save(f"phi{step}.png")

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
                
                params['cur_env_id'] = id(str(env.unwrapped.get_cur_env()))
                break

if __name__ == '__main__':
    main()