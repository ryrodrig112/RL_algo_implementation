class Agent:
    def __init__(self, env):
        self.action_size = env.action_space.n
        print("Action size:", self.action_size)

    def get_action(self, state):
        #         action = random.choice(range(self.action_size))
        pole_angle = state[2]
        action = 0 if pole_angle < 0 else 1
        return action