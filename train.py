import numpy as np
import random
import pickle
from collections import defaultdict
from env.platformer_env import PlatformerEnv

from main import LEVELS  # reuse levels

level = random.choice(LEVELS)
env = PlatformerEnv(level)

q_table = defaultdict(lambda: np.zeros(env.action_space.n))

alpha = 0.2
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.05

episodes = 10000

for episode in range(episodes):
    state, _ = env.reset()
    state = state = tuple(state)

    done = False

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state = tuple(next_state)

        q_table[state][action] += alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
        )

        state = next_state

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

print("Training complete")

# SAVE
with open("q_table.pkl", "wb") as f:
    pickle.dump(dict(q_table), f)