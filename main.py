import random

import numpy as np
import gymnasium as gym

import atari_env

def main():
    env_ids = ['ALE/Pong-v5','ALE/Breakout-v5','ALE/Centipede-v5','ALE/SpaceInvaders-v5','ALE/Asteroids-v5']
    with atari_env.make_env(env_ids, render_mode='human') as env:
        env.reset()
        for i in range(200):
            a = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(a)
            print(reward)

if __name__ == '__main__':
    main()
