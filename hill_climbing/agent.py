import numpy
import random
import numpy as np


class Agent:
    def __init__(self, env):
        self.state_dim = env.observation_space.shape
        self.action_size = env.action_space.n
        self.noise_amplitude = 1e-2
        self.init_model()

    def init_model(self):
        self.current_weights = 1e-4 * np.random.rand(*self.state_dim, self.action_size)
        self.best_weights = self.current_weights
        self.best_perf = -np.Inf

    def get_action(self, state):
        expectation_vector = state @ self.current_weights
        action = np.argmax(expectation_vector)
        return action

    def update_weights(self, ep_perf):
        if ep_perf >= self.best_perf:
            self.best_perf = ep_perf
            self.best_weights = self.current_weights
            self.noise_amplitude = max(self.noise_amplitude / 2, 1e-3)
        else:
            self.noise_amplitude = min(self.noise_amplitude * 2, 2)
        self.current_weights = self.best_weights + self.noise_amplitude * np.random.rand(*self.state_dim,
                                                                                         self.action_size)
