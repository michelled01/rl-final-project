import random

import numpy as np
import gymnasium as gym

def get_env_specs_ale_all():
    return [
        spec for spec in gym.envs.registry.values()
        if spec.namespace == 'ALE' and spec.kwargs.get('obs_type') == 'rgb']

def get_env_specs_ale_filtered():
    def criteria(spec):
        env = gym.make(spec) # this is slow (needs to load ROM)
        print(spec.name, env.action_space)
        return env.action_space == gym.spaces.Discrete(4) # TODO
    return [spec for spec in get_env_specs_ale_all() if criteria(spec)]

def main():
    env_ids = ['ALE/Pong-v5','ALE/Breakout-v5','ALE/Centipede-v5','ALE/SpaceInvaders-v5','ALE/Asteroids-v5']
    # thunk is necessary to capture env_id by value
    env_thunk = lambda env_id: lambda: gym.make(env_id, full_action_space=True, render_mode='human') # render_mode=None to disable rendering
    with gym.vector.AsyncVectorEnv([env_thunk(env_id) for env_id in env_ids]) as env:
        env.reset()
        for i in range(200):
            a = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(a)
            # print(reward)
            #print(f"obs = {obs}\nreward = {reward}\nterminated = {terminated}\ntruncated = {truncated}, info = {info}")

if __name__ == '__main__':
    from pprint import pprint
    pprint(get_env_specs_ale_all())
    main()
