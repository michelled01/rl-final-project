import numpy as np
import gym
import random
import matplotlib.pyplot as plt
import os

def test_pong():

    env = gym.make('ALE/Pong-v5',render_mode='human')

    """Reset our environment, notice it returns the first frame of the game"""
    first_frame = env.reset()

    """ In Pong the actions are:
    0 = Stay Still
    1 = Shoot Ball
    2 = Move Right
    3 = Move Left
    4 = Move Right and Shoot Ball
    5 = Move Left and Shoot Ball
    """

    """Now lets take a bunch of random actions and watch the gameplay using render.
    If the game ends we will reset it using env.reset"""

    for i in range(10000):
        a = random.sample([0,1,2,3,4,5] , 1)[0]
        f_p,r,d,_,info = env.step(a)
        env.render()
        if d == True:
            env.reset()

    env.close()

if __name__ == "__main__":
    test_pong()
