import stable_baselines3

import os

def main():
    baseline_algo = stable_baselines3.A2C # or A2C, try getting DQN to work too
    atari_name = 'Pong'
    model_path = f"baseline_{baseline_algo.__name__}_{atari_name}"
    env = stable_baselines3.common.env_util.make_atari_env(f"ALE/{atari_name}-v5")
    if not os.path.isfile(f"{model_path}.zip"):
        model = baseline_algo("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10000, log_interval=4)
        model.save(model_path)
        del model # remove to demonstrate saving and loading

    model = baseline_algo.load(model_path)

    obs = env.reset()
    while True:
      action, _states = model.predict(obs, deterministic=False)
      obs, reward, dones, info = env.step(action)
      env.render('human')
      import time
      time.sleep(1/60)

if __name__ == '__main__':
    main()
