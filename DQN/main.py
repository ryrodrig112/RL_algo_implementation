# This is a sample Python script.
import gym
import agent

env_name = "CartPole-v1"
env = gym.make(env_name, render_mode = 'human')
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

agent = agent.Agent(env)
state = env.reset()

best_perf = 0
ep_perf = 0
# while best_perf < 200:

for _ in range(200):
    action = agent.get_action(state)
    state, step_reward, done, info = env.step(action)
    ep_perf += step_reward

if done:
    print(ep_perf)
    env.reset()
    ep_perf = 0

# if ep_perf > best_perf:
#     best_perf = ep_perf