import numpy

class Agent:
    def __init__(self, env):
        self.action_size = env.action_space.n
        self.weights = ...

    def get_action(self, state):
        pole_angle = state[2]
        action = 0 if pole_angle < 0 else 1
        return action

    def get_update_weights(self, state):

        new_weights = ...
        self.weights = new_weights
