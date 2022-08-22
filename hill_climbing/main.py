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
    agent = hc_agent.VectorAgent(envs, gym_id)
    states = envs.reset()
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    assert envs.observation_space.shape[0] == num_envs
    print('Num envs: ', num_envs)
    print('Env input size: ', envs.observation_space.shape[1])
    print('Env action space size: ', envs.single_action_space.n)
    print('')
    episodes_finished = 0
    global_step = 0
    completions = 0

    while agent.last_five_mean_perf < 500:
        global_step += 1
        initial_states = states
        actions = agent.get_action(states)
        states, rewards, dones, infos = envs.step(actions)
        for state in range(initial_states.shape[0]): #check to make sure no actions in any env are just duplicating the previous state
            assert not (initial_states[state] == states[state]).all()

        if infos:
            for env in range(len(infos['episode'])):
                if infos['episode'][env]: # non terminal envs are None, while terminal environments are stored in the information wrapper
                    episodes_finished +=1
                    ep_perf = infos['episode'][env]['r']
                    assert ep_perf <= global_step

                    agent.update_weights(ep_perf)
                    agent.update_history(ep_perf,episodes_finished, global_step)
    print(f"Tuning Completed - {episodes_finished} required")
    # print('Running Additional Eps')
    # additional_eps = 0
    # while additional_eps < tuning_steps/3: # these are just steps still lol
    #     global_step += 1
    #     initial_states = states
    #     actions = agent.get_action(states)
    #     states, rewards, dones, infos = envs.step(actions)
    #     for state in range(initial_states.shape[
    #                            0]):  # check to make sure no actions in any env are just duplicating the previous state
    #         assert not (initial_states[state] == states[state]).all()
    #
    #     if infos:
    #         for env in range(len(infos['episode'])):
    #             if infos['episode'][env]:
    #                 # non terminal envs are None, while terminal environments are stored in the information wrapper
    #                 episodes_finished += 1
    #                 additional_eps += 1
    #                 ep_perf = infos['episode'][env]['r']
    #                 assert ep_perf <= global_step
    #
    #                 agent.update_weights(ep_perf)
    #                 agent.update_history(ep_perf, episodes_finished, global_step)

    print('')
    agent.show()
    print('')
    print(f"Steps required to converge: {global_step}")
    print(f"Episodes required to converge: {episodes_finished}")

    train_df = pd.DataFrame(
        data={'number_of_episodes': agent.episode_history,
              'weight_history' : agent.weight_history,
              'noise_history': agent.noise_history,
              'number_of_steps': agent.step_history,
              'perf_tracker': agent.perf_history})
    train_df.to_csv('trial_history.csv')



    print("Begin Testing")
    # Test Trained Agent
    test_episodes = 0
    env = gym.make(gym_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = gym.wrappers.RecordVideo
    agent = hc_agent.SingleAgent(""
                                 "")
    state = env.reset()
    reward_tracker = []
    while test_episodes < 100:
        expectation_vector = state @ optimal_weights
        action = np.argmax(expectation_vector)
        state, reward, done, info = env.step(action)
        if done:
            env.reset()
            test_episodes += 1
            episode_reward = info['episode']['r']
            reward_tracker.append(episode_reward)

    success_pct = sum([ep == 500 for ep in reward_tracker])
    print(reward_tracker[:20])
    print(success_pct)



