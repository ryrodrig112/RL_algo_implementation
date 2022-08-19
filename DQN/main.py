# This is a sample Python script.
import gym
import agent

env_name = "CartPole-v1"
env = gym.make(env_name)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

agent = agent.Agent(env)
state = env.reset()

for _ in range(200):
    action = env.action_space.sample()
    action = agent.get_action(state)
    state, reward, done, info = env.step(action)
    env.render()