import numpy
import random
import numpy as np


class Agent:
    def __init__(self, gym_id):
        self.gym_id = gym_id
        self.state_dim = envs.observation_space.shape
        self.num_envs = envs.observation_space.shape[0]
        self.num_inputs = self.state_dim[1]
        self.action_size = envs.single_action_space.n
        self.max_score = 500

    def show(self):
        print(f"Gym type: {self.gym_id}")
        print(f"Num envs: {self.num_envs}")
        print(f"Num inputs: {self.num_inputs}")
        print(f"Num potential actions: {self.action_size}")
        print(f"Max Possible Score: {self.max_score}")
        print(f"Best Performance: {self.best_perf}")

    def update_history(self, ep_perf, episode, global_step):
        self.noise_history.append(self.noise_amplitude)
        self.weight_history.append(self.current_weights)
        self.perf_history.append(ep_perf)
        self.episode_history.append(episode)
        self.step_history.append(global_step)
        if self.perf_history:
            if len(self.perf_history) > 5:
                self.last_five_mean_perf = np.mean(self.perf_history[-5:])


class VectorAgent(Agent):
    def __init__(self, envs, gym_id):
        self.gym_id = gym_id
        self.state_dim = envs.observation_space.shape
        self.num_envs = envs.observation_space.shape[0]
        self.num_inputs = self.state_dim[1]
        self.action_size = envs.single_action_space.n

    def init_model(self):
        self.current_weights = 1e-4 * np.random.rand(self.num_inputs, self.action_size)
        self.noise_amplitude = 1e-2
        self.best_weights = self.current_weights
        self.best_perf = -np.Inf
        self.eps_since_restart = 0

    def get_action(self, states):
        expectation_vectors = [state @ self.current_weights for state in states]
        actions = [np.argmax(vector) for vector in expectation_vectors]
        return actions

    def update_weights(self, ep_perf):
        # if ep_perf >= best_perf: replace best weights and halve noise
        # if ep_perf < best_perf: keep best weights and double noise amplitude
        # if we have scored max for the first time, lower noise to set amount
        # if we scored it consecutively, begin halving
        if ep_perf >= self.best_perf:
            self.best_perf = ep_perf
            self.best_weights = self.current_weights
            self.noise_amplitude = max(self.noise_amplitude / 2, 1e-4)
        else:
            self.noise_amplitude = min(self.noise_amplitude * 2, 2)
        self.current_weights = self.best_weights + (self.noise_amplitude * np.random.rand(self.num_inputs,
                                                                                              self.action_size))


class SingleAgent(Agent):
    def __init__(self, env, gym_id):
        self.state_dim = env.observation_space.shape
        self.action_size = envs.single_action_space.n

    def get_action(self, state):
        expectation_vectors = state @ self.current_weights
        action = np.argmax(expectation_vectors)
        return action

    def update_weights(self, ep_perf):
        # if ep_perf >= best_perf: replace best weights and halve noise
        # if ep_perf < best_perf: keep best weights and double noise amplitude
        # if we have scored max for the first time, lower noise to set amount
        # if we scored it consecutively, begin halving
        if ep_perf >= self.best_perf:
            self.best_perf = ep_perf
            self.best_weights = self.current_weights
            self.noise_amplitude = max(self.noise_amplitude / 2, 1e-4)
        else:
            self.noise_amplitude = min(self.noise_amplitude * 2, 2)
        self.current_weights = self.best_weights + (self.noise_amplitude * np.random.rand(self.num_inputs,
                                                                                 self.action_size))

    def get_optimal_weights(self):
        maxes = []
        for i in range(len(self.perf_history)):
            if self.best_perf[i] == self.max_score:
                maxes.append(i)
        best_weights = self.perf_history([maxes])
        optimal_weights = np.mean(best_weights)
        return optimal_weights
