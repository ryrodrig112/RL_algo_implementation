import gym
import agent

env_name = "CartPole-v1"
env = gym.make(env_name, render_mode = 'human')
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

agent = agent.Agent(env)
state = env.reset()

best_perf = 0
attempts = 0

num_episodes = 100

for episode in range(num_episodes):
    ep_perf = 0
    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        state, step_reward, done, info = env.step(action)
        ep_perf += step_reward
    if ep_perf > best_perf:
        best_perf = ep_perf
    print("Attempt: {}, Performance: {}, Best Performance:{}, Noise: {}".format(episode, ep_perf, best_perf, agent.noise_amplitude))
    agent.update_weights(ep_perf)


