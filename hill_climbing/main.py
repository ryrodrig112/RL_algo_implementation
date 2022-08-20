import gym
import hc_agent
import pandas as pd
import numpy as np


if __name__ == "__main__":
    envs = gym.vector.make("CartPole-v1", num_envs=3)
    envs.reset()
    actions = np.array([1, 0, 1])
    observations, rewards, dones, infos = envs.step(actions)

    print("observations: {}".format(observations))
    print("rewards: {}".format(rewards))
    print("dones: {}".format(dones))
    print("infos: {}".format(infos))


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
