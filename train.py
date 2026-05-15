import numpy as np
import random
import pickle
from collections import defaultdict
from env.platformer_env import PlatformerEnv
from main import LEVELS

temp_env = PlatformerEnv(LEVELS[0])
q_table = defaultdict(lambda: np.zeros(temp_env.action_space.n))

alpha = 0.2
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
episodes_per_level = 200

for level_index, level in enumerate(LEVELS):
    print(f"Training level {level_index + 1}")
    env = PlatformerEnv(level)

    for episode in range(episodes_per_level):
        state, _ = env.reset()
        state = tuple(state)

        done = False

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = tuple(next_state)

            # Q-learning update
            q_table[state][action] += alpha * (
                reward
                + gamma * np.max(q_table[next_state])
                - q_table[state][action]
            )
            state = next_state

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if episode % 50 == 0:
            print(
                f"Episode {episode} | "
                f"Epsilon {epsilon:.3f}"
            )

print("Training complete")

with open("q_table.pkl", "wb") as f:
    pickle.dump(dict(q_table), f)