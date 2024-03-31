from stable_baselines3 import DQN

import atari_env

def main():
    env_ids = ['ALE/Pong-v5','ALE/Breakout-v5','ALE/Centipede-v5','ALE/SpaceInvaders-v5','ALE/Asteroids-v5']
    # thunk is necessary to capture env_id by value
    with atari_env.make_env(env_ids) as env:
        model = DQN("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10000, log_interval=4)
        model.save("dqn_Pong_Breakout_Centipede_SpaceInvaders_Asteroids")

        del model # remove to demonstrate saving and loading

        model = DQN.load("dqn_Pong_Breakout_Centipede_SpaceInvaders_Asteroids")

        obs, info = env.reset()
        while True:
          action, _states = model.predict(obs, deterministic=True)
          obs, reward, terminated, truncated, info = env.step(action)
          if terminated or truncated:
            obs, info = env.reset()

if __name__ == '__main__':
    main()
