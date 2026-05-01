import random

class RandomAgent:
    def act(self, state):
        return random.randint(0, 2)
