import gym


def make_env(gym_id):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk