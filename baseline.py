import time
import stable_baselines3
import tensorflow as tf

import os

def main():
    # atari example: https://stable-baselines.readthedocs.io/en/master/guide/examples.html#id2
    ATARI_NAME = 'Pong'
    BASELINE_ALGO = stable_baselines3.PPO # or PPO or A2C, try getting DQN to work too
    TOTAL_TIMESTEPS = 1_000_000
    MODEL_PATH = f"baseline_{ATARI_NAME}_{BASELINE_ALGO.__name__}_{TOTAL_TIMESTEPS}"
    LOG_DIR = f"logs/{MODEL_PATH}/"
    FRAME_TIME = 1.0 / 24.0

    env = stable_baselines3.common.env_util.make_atari_env(f"ALE/{ATARI_NAME}-v5")
    env = stable_baselines3.common.vec_env.VecFrameStack(env, n_stack=4)

    if not os.path.isfile(f"{MODEL_PATH}.zip"):
        model = BASELINE_ALGO("CnnPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        model.save(MODEL_PATH)
        del model # remove to demonstrate saving and loading

    model = BASELINE_ALGO.load(MODEL_PATH)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, reward, dones, info = env.step(action)
        env.render('human')
        time.sleep(FRAME_TIME)
        if dones: break

if __name__ == '__main__':
    main()
