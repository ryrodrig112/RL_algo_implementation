import gym
import agent

env_name = "CartPole-v1"
env = gym.make(env_name, render_mode = 'human')
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

best_perf = 0
attempts = 0

num_trials = 100
num_episodes = 100
goal_consecutive_max = 3
num_consecutive_max = 0

num_eps_in_trial = []
trial_weights_tracker = []
trial_noise_tracker = []
trial_perf_tracker = []


for trials in range(num_trials):
    agent = agent.Agent(env)
    for episode in range(num_episodes):
        ep_perf = 0
        state = env.reset()
        done = False
        while not done: # agent applies policy to environment step, takes next step, until episode ends
            action = agent.get_action(state)
            state, step_reward, done, info = env.step(action)
            ep_perf += step_reward
        # Once the episode ends...
        # Update tracking metrics (perfs, best_perf, num_consecutive_max)
        # Print performance
        # Update agent
        agent.update_history(ep_perf)
        if ep_perf == 500:
            num_consecutive_max += 1
            if num_consecutive_max > goal_consecutive_max:
                break
        else:
            num_consecutive_max = 0
        print("Attempt: {}, Performance: {}, Best Performance:{}, Consecutive 500's:{}, Noise: {}"
              .format(episode, ep_perf, agent.best_perf, num_consecutive_max,agent.noise_amplitude))
        agent.update_weights(ep_perf)
    num_eps_in_trial.append(episode)
    trial_weights_tracker.append(agent.weight_history)
    trial_noise_tracker.append(agent.noise_history)
    trial_reward_tracker.append(agent.perf_history)





