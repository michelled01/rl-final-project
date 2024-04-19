import stable_baselines3
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

import os

def main():

    # atari example: https://stable-baselines.readthedocs.io/en/master/guide/examples.html#id2
    baseline_algo = stable_baselines3.PPO # or A2C, try getting DQN to work too
    atari_name = 'Pong'
    model_path = f"baseline_{baseline_algo.__name__}_{atari_name}"
    env = stable_baselines3.common.env_util.make_atari_env(f"ALE/{atari_name}-v5")
    env = stable_baselines3.common.vec_env.VecFrameStack(env, n_stack=4)
    
    log_dir = f"logs/{model_path}"
    tensorboard_callback = TensorBoard(log_dir=log_dir)

    if not os.path.isfile(f"{model_path}.zip"):
        model = baseline_algo("CnnPolicy", env, verbose=1)
        model.learn(total_timesteps=20_000, log_interval=4, callback=tensorboard_callback)
        model.save(model_path)
        del model # remove to demonstrate saving and loading

    model = baseline_algo.load(model_path)

    obs = env.reset()
    step = 0
    while True:
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, dones, info = env.step(action)
        env.render('human')
        import time
        time.sleep(1/60)

        with tf.summary.create_file_writer(log_dir).as_default():
            tf.summary.scalar('reward', reward.item() ,step=step)
            tf.summary.scalar('episode_length', info[0]['episode_frame_number'],step=step)
        
        tensorboard_callback.on_epoch_end(step)
        step += 1

        if dones: break

if __name__ == '__main__':
    main()
