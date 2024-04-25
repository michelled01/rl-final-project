import gymnasium as gym
import numpy as np

def get_env_specs_ale_all():
    return [
        spec for spec in gym.envs.registry.values()
        if spec.namespace == 'ALE' and spec.kwargs.get('obs_type') == 'rgb']

'''
def get_env_specs_ale_filtered():
    def criteria(spec):
        env = gym.make(spec) # this is slow (needs to load ROM)
        print(spec.name, env.action_space)
        return env.action_space == gym.spaces.Discrete(4) # TODO
    return [spec for spec in get_env_specs_ale_all() if criteria(spec)]
'''

'''
def make_env(env_ids, *, render_mode=None):
    # thunk is necessary to capture env_id by value
    env_thunk = lambda env_id: lambda: gym.make(env_id, full_action_space=True, render_mode=render_mode) # None to disable rendering, 'human' to enable
    return gym.vector.AsyncVectorEnv([env_thunk(env_id) for env_id in env_ids])
'''

class RandomAtariEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 60}
    action_space = gym.spaces.Discrete(18)
    observation_space = gym.spaces.Box(0, 255, (84, 84, 1), np.uint8)

    def __init__(self, *,
        env_names: list[str],
        max_episode_steps: int,
        render_mode: str | None = None,
    ):
        assert render_mode is None or render_mode in self.metadata['render_modes']
        env_names = sorted(env_names)
        self.render_mode = render_mode
        self._envs = [
            gym.wrappers.TimeLimit(
                gym.wrappers.AtariPreprocessing(
                    gym.make(
                        f"ALE/{env_name}-v5",
                        frameskip=1,
                        full_action_space=True,
                        render_mode=render_mode,
                    ),
                    grayscale_newaxis=True
                ),
                max_episode_steps=max_episode_steps
            )
            for env_name in env_names
        ]
        self._env_step_counts = np.zeros(len(self._envs), dtype=int)
        self._cur_env_idx = None

    def reset(self, *args, **kwargs):
        self._cur_env_idx = self._env_step_counts.argmin()
        #self._cur_env_idx = self.np_random.choice(len(self._envs), p=self._env_weights)
        return self._cur_env.reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        self._env_step_counts[self._cur_env_idx] += 1
        return self._cur_env.step(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self._cur_env.render(*args, **kwargs)

    def close(self, *args, **kwargs):
        for env in self._envs:
            env.close(*args, **kwargs)

    @property
    def _cur_env(self):
        return None if self._cur_env_idx is None else self._envs[self._cur_env_idx]

if __name__ == '__main__':
    from pprint import pprint
    pprint(get_env_specs_ale_all())
