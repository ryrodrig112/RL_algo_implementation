import gym
import environments
import hc_agent
import pandas as pd
import numpy as np
import json

# Define Parameters
with open("../params.json", "r") as jsonfile:
    params = json.load(jsonfile)
    print("Read successful")

gym_id = params['training_params']['gym_id']  # Training Params
num_envs = params['training_params']['num_envs']
num_additional_train_eps = params['training_params']['additional_eps']
num_test_episodes = params['test_params']['num_eps'] # Test Params

if __name__ == "__main__":
    envs = gym.vector.SyncVectorEnv(
        [environments.make_env(gym_id) for i in range(num_envs)]
    )
    training_agent = hc_agent.VectorAgent(envs, gym_id)
    states = envs.reset()
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    assert envs.observation_space.shape[0] == num_envs
    print('Num envs: ', num_envs)
    print('Env input size: ', envs.observation_space.shape[1])
    print('Env action space size: ', envs.single_action_space.n)
    print('')
    episodes_finished = 0
    global_step = 0

    while training_agent.last_five_mean_perf < 500:
        global_step += 1
        initial_states = states
        actions = training_agent.get_action(states)
        states, rewards, dones, infos = envs.step(actions)
        for state in range(initial_states.shape[0]): #check to make sure no actions in any env are just duplicating the previous state
            assert not (initial_states[state] == states[state]).all()

        if infos:
            for env in range(len(infos['episode'])):
                if infos['episode'][env]: # non terminal envs are None, while terminal environments are stored in the information wrapper
                    episodes_finished +=1
                    ep_perf = infos['episode'][env]['r']
                    assert ep_perf <= global_step

                    training_agent.update_weights(ep_perf)
                    training_agent.update_history(ep_perf,episodes_finished, global_step)

    tuning_eps = episodes_finished
    print(f"Initial Tuning Completed: {tuning_eps} eps required")
    print('Running Additional Eps')
    additional_eps = 0

    while additional_eps < num_additional_train_eps:
        global_step += 1
        initial_states = states
        actions = training_agent.get_action(states)
        states, rewards, dones, infos = envs.step(actions)
        for state in range(initial_states.shape[
                               0]):  # check to make sure no actions in any env are just duplicating the previous state
            assert not (initial_states[state] == states[state]).all()
        if infos:
            for env in range(len(infos['episode'])):
                if infos['episode'][env]:
                    # non terminal envs are None, while terminal environments are stored in the information wrapper
                    episodes_finished += 1 # includes pre-convergence eps
                    additional_eps += 1 # only includes additional episodes
                    ep_perf = infos['episode'][env]['r']
                    assert ep_perf <= global_step
                    training_agent.update_weights(ep_perf)
                    training_agent.update_history(ep_perf, episodes_finished, global_step)

    print('')
    print('Training Complete')
    print(f"Total Training Episodes: {episodes_finished}")

    train_df = pd.DataFrame(
        data={'number_of_episodes': training_agent.episode_history,
              'weight_history' : training_agent.weight_history,
              'noise_history': training_agent.noise_history,
              'number_of_steps': training_agent.step_history,
              'perf_tracker': training_agent.perf_history})
    train_df.to_csv('train_df.csv', index=False)

    # Test Trained Agent
    print("Begin Testing")
    test_weights = training_agent.get_optimal_weights()
    test_episode = 0
    env = gym.make(gym_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    test_agent = hc_agent.SingleAgent(env, gym_id)
    test_agent.set_weights(test_weights)
    state = env.reset()
    reward_tracker = []
    while test_episode < num_test_episodes:
        action = test_agent.get_action(state)
        state, reward, done, info = env.step(action)
        if done:
            env.reset()
            test_episode += 1
            episode_reward = info['episode']['r']
            reward_tracker.append(episode_reward)

    success_tracker = [ep_perf == 500 for ep_perf in reward_tracker]
    success_pct = (sum(success_tracker)/num_test_episodes) * 100
    print(f"Agent Success Ratio: {success_pct}%")

    test_df = pd.DataFrame(
        data={'episode': np.arange(1, num_test_episodes+1),
              'perf_tracker': reward_tracker,
              'success_tracker': success_tracker
              })
    
    test_df.to_csv('test_df.csv', index=False)





