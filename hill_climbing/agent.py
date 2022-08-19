import numpy
import random
import numpy as np


class Agent:
    def __init__(self, env):
        self.state_dim = env.observation_space.shape
        self.action_size = env.action_space.n
        self.noise_amplitude = 1e-2
        self.init_model()
        self.max_score = 500
        self.perf_history = []
        self.noise_history = []
        self.weight_history = []

    def init_model(self):
        self.current_weights = 1e-4 * np.random.rand(*self.state_dim, self.action_size)
        self.best_weights = self.current_weights
        self.best_perf = -np.Inf

    def get_action(self, state):
        expectation_vector = state @ self.current_weights
        action = np.argmax(expectation_vector)
        return action

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
            self.noise_amplitude = max(self.noise_amplitude / 2, 1e-6)
            if ep_perf == self.max_score:
                self.noise_amplitude = self.noise_amplitude / 4
        else:
            self.noise_amplitude = min(self.noise_amplitude * 2, 2)
        self.current_weights = self.best_weights + (self.noise_amplitude * np.random.rand(*self.state_dim,
                                                                                         self.action_size))

