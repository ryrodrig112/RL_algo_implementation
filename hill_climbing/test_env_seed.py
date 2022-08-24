import gym
import hc_agent
import pandas as pd
import numpy as np


def make_envs(gym_id, seed):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# set global params - could be in args(?)
gym_id = "CartPole-v1"
num_envs = 10
seed = 1


for i in range(10):
    env = gym.make(gym_id)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    state = env.reset()
    print(i, state)