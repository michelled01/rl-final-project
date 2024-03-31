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

def make_env(env_ids, *, render_mode=None):
    # thunk is necessary to capture env_id by value
    env_thunk = lambda env_id: lambda: gym.make(env_id, full_action_space=True, render_mode=render_mode) # None to disable rendering, 'human' to enable
    return gym.vector.AsyncVectorEnv([env_thunk(env_id) for env_id in env_ids])

if __name__ == '__main__':
    from pprint import pprint
    pprint(get_env_specs_ale_all())
