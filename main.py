import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import random
import matplotlib.pyplot as plt
import os

def test_random_env():
    Envs = ['ALE/Pong-v5',"ALE/Breakout-v5","ALE/Centipede-v5","ALE/SpaceInvaders-v5","ALE/Asteroids-v5"]
    env = gym.make(random.choice(Envs), render_mode='human')
    env.reset()
    
    for i in range(200):
        print(i)
        a = env.action_space.sample()
        f_p,r,d,t,info = env.step(a)
        env.render()
        if d or t:
            env = gym.make(random.choice(Envs), render_mode='human')
            env.reset()

    env.close()

if __name__ == "__main__":
    test_random_env()
