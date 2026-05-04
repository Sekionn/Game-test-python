import numpy as np

class QLearningAgent:
    def __init__(self, q_table, action_size, epsilon=0.1):
        self.q = q_table
        self.action_size = action_size
        self.epsilon = epsilon

    def act(self, state):
        state = tuple(state)

        if state not in self.q:
            self.q[state] = np.zeros(self.action_size)

        # exploration
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)

        return np.argmax(self.q[state])