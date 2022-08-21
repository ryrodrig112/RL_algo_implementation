import gym
import hc_agent
import pandas as pd
import numpy as np


def make_env(gym_id, seed):
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
num_envs = 6
seed = 1

if __name__ == "__main__":
    envs = gym.vector.SyncVectorEnv(
        [make_env(gym_id, seed + i) for i in range(num_envs)]
    )
    agent = hc_agent.Agent(envs, gym_id)
    states = envs.reset()
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    assert envs.observation_space.shape[0] == num_envs
    print('Num envs: ', num_envs)
    print('Env input size: ', envs.observation_space.shape[1])
    print('Env action space size: ', envs.single_action_space.n)

    for step in range(200):
        actions = agent.get_action(states)
        states, rewards, dones, infos = envs.step(actions)

        print(states)

        # if infos:
        #     for env in range(len(infos['episode'])):
        #         if infos['episode'][env]: #nonterminal envs are None, while terminal environments are stored in the information wrapper
        #             print(f"step: {step}: , env: {env}, episodic return: {infos['episode'][env]['r']}")




# print("Observation space:", env.observation_space)
# print("Action space:", env.action_space)
# t0 = time.time()
#
# while num_consecutive_max < goal_consecutive_max:
#     episode += 1
#     ep_perf = 0
#     state = env.reset()
#     done = False
#     while not done:  # agent applies policy to environment step, takes next step, until episode ends
#         action = agent.get_action(state)
#         state, step_reward, done, info = env.step(action)
#         ep_perf += step_reward
#     # Once the episode ends...
#     # Update tracking metrics (perfs, best_perf, num_consecutive_max)
#     # Print performance
#     # Update agent
#     agent.update_history(ep_perf)
#     if ep_perf == 500:
#         num_consecutive_max += 1
#     else:
#         num_consecutive_max = 0
#     print("Trial: {}: Attempt: {}, Performance: {}, Best Performance:{}, Consecutive 500's:{}, Noise: {}"
#           .format(trial, episode, ep_perf, agent.best_perf, num_consecutive_max, agent.noise_amplitude))
#     agent.update_weights(ep_perf)
# t1 = time.time()
# print("Time elapsed for 3 trials: {}".format(t1 - t0))
# trial_df = pd.DataFrame(
#     data={'number_of_episodes': num_eps_in_trial,
#           'noise_history': trial_noise_tracker,
#           'perf_tracker': trial_perf_tracker})
# trial_df.to_csv('trial_history.csv')
