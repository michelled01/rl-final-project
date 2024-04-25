import multitask_atari

import stable_baselines3 as sb3, stable_baselines3.common.env_checker
import gymnasium as gym

import os, random, warnings, pathlib

# atari example : https://stable-baselines.readthedocs.io/en/master/guide/examples.html#id2

config = {
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

def main():
    if 'seed' in config:
        import random, numpy as np, torch
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
    else:
        warnings.warn('random number generators have not been seeded')

    algo = getattr(sb3, config['algo'])
    path = pathlib.Path('baselines') / f"{'-'.join(sorted(config['atari-env-names']))}_{config['algo']}_{config['total-timesteps']}"
    path.mkdir(parents=True, exist_ok=True)
    log_dir = path / 'logs'
    model_path = path / 'model.zip'

    env = gym.make('multitask-atari', max_episode_steps=config['max-episode-steps'], env_names=config['atari-env-names'], render_mode='rgb_array')
    sb3.common.env_checker.check_env(env.unwrapped, warn=True, skip_render_check=False)

    if not model_path.is_file():
        model = algo(config['policy'], env, verbose=1, tensorboard_log=log_dir)
        model.learn(total_timesteps=config['total-timesteps'])
        model.save(model_path)
        del model # remove to demonstrate saving and loading

    env = gym.wrappers.HumanRendering(env)
    model = algo.load(model_path)

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if reward:
            print(reward)
        if terminated or truncated:
            env.reset()

if __name__ == '__main__':
    main()
