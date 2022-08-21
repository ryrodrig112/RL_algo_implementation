import numpy
import random
import numpy as np


class Agent:
    def __init__(self, envs, gym_id):
        self.gym_id = gym_id
        self.state_dim = envs.observation_space.shape
        self.num_envs = envs.observation_space.shape[0]
        self.num_inputs = self.state_dim[1]
        self.action_size = envs.single_action_space.n
        self.noise_amplitude = 1e-2
        self.init_model()
        self.max_score = 500
        self.perf_history = []
        self.noise_history = []
        self.weight_history = []

    def show(self):
        print(f"Gym type: {self.gym_id}")
        print(f"Num envs: {self.num_envs}")
        print(f"Num inputs: {self.num_inputs}")
        print(f"Num potential actions: {self.action_size}")
        print(f"Max Possible Score: {self.max_score}")
        print(f"Max Best Performance: {self.best_perf}")


    def init_model(self):
        self.current_weights = 1e-4 * np.random.rand(self.num_inputs, self.action_size)
        self.best_weights = self.current_weights
        self.best_perf = -np.Inf

    def get_action(self, states):
        expectation_vectors = [state @ self.current_weights for state in states]
        actions = [np.argmax(vector) for vector in expectation_vectors]
        return actions


    def update_history(self, ep_perf):
        self.noise_history.append(self.noise_amplitude)
        self.weight_history.append(self.current_weights)
        self.perf_history.append(ep_perf)

    def update_weights(self, ep_perf):
        # if ep_perf >= best_perf: replace best weights and halve noise
        # if ep_perf < best_perf: keep best weights and double noise amplitude
        # if we have scored max for the first time, lower noise to set amount
        # if we scored it consecutively, begin halving
        if ep_perf >= self.best_perf:
            self.best_perf = ep_perf
            self.best_weights = self.current_weights
            self.noise_amplitude = max(self.noise_amplitude / 1.5, 1e-6)
            if ep_perf == self.max_score:
                self.noise_amplitude = self.noise_amplitude / 4
        else:
            if self.best_perf == self.max_score:
                self.noise_amplitude == 1e-2
            else:
                self.noise_amplitude = min(self.noise_amplitude * 1.5, 2)
        self.current_weights = self.best_weights + (self.noise_amplitude * np.random.rand(*self.num_inputs,
                                                                                         self.action_size))

