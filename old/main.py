import random

import numpy as np
import gymnasium as gym

import atari_env

def main():
    env_ids = [
      'ALE/Boxing-v5', # 1707%
      'ALE/Breakout-v5', # 1327%
      # 'ALE/Pong-v5', # 132%
      # 'ALE/SpaceInvaders-v5', # 121%
      # 'ALE/Centipede-v5', # 62%
      # "ALE/Asteroids-v5' # 7%
    ]
    with atari_env.make_env(env_ids, render_mode='human') as env:
        env.reset()
        for i in range(200):
            a = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(a)
            print(reward)
            #print(f"obs = {obs}\nreward = {reward}\nterminated = {terminated}\ntruncated = {truncated}, info = {info}")

if __name__ == '__main__':
    main()
