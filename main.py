import pygame
from env.platformer_env import PlatformerEnv
from agents.random_agent import RandomAgent
from agents.q_learning_agent import QLearningAgent
import pickle
import numpy as np

LEVELS = [
    # Level 1
    [
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,9,0,0,0,0,0,0,0,0,0,0,0,0,3,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    ],
    #Level 2
    [
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,9,0,0,0,0,0,0,0,0,0,0,0,0,3,1,1],
        [1,1,9,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    ],
    #Level 3
    [
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1],
        [1,1,9,0,0,0,0,0,0,0,0,0,1,0,0,3,1,1],
        [1,1,1,1,1,4,0,0,0,4,1,5,1,5,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    ],
    #Level 4
    [
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,9,0,0,0,0,0,0,0,0,0,0,0,0,3,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    ],
    #Level 5
    [
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,9,0,0,0,0,0,0,0,0,0,0,0,0,3,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    ],

]

MODE = "ai"


def run(mode):
    pygame.init()
    level_index = 2
    env = PlatformerEnv(LEVELS[level_index], render=True)
    state, _ = env.reset()

    clock = pygame.time.Clock()

    agent = None
    q_table = None

    if mode == "ai":
        with open("q_table.pkl", "rb") as f:
            q_table = pickle.load(f)

        agent = QLearningAgent(q_table, env.action_space.n)

    running = True
    while running:
        action = 2
        vertical_input = 0  # -1 up, +1 down, 0 none

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if mode == "human":
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action = 0
            elif keys[pygame.K_RIGHT]:
                action = 1
            elif keys[pygame.K_UP]:
                action = 3
                vertical_input = 1
            elif keys[pygame.K_DOWN]:
                action = 4
                vertical_input = 0
        else:
            action = agent.act(state)

        state, reward, terminated, truncated, _ = env.step(action, vertical_input)
        done = terminated or truncated

        if done:
            print(f"Completed level {level_index + 1}")
            level_index += 1

            if level_index >= len(LEVELS):
                print("All levels completed!")
                running = False
                continue

            env = PlatformerEnv(LEVELS[level_index], render=True)
            state, _ = env.reset()

        clock.tick(10)

    pygame.quit()


if __name__ == "__main__":
    run(MODE)