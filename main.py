import sys

import pygame
from env.platformer_env import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_NONE,
    ACTION_RIGHT,
    ACTION_UP,
    PlatformerEnv,
)
from agents.random_agent import RandomAgent

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

MODE = "human"

def run(mode):
    level_index = 0
    env = PlatformerEnv(
        LEVELS[level_index],
        render=True,
        run_label=mode,
        level_name=f"level_{level_index + 1}",
    )
    state, info = env.reset()

    clock = pygame.time.Clock()
    agent = RandomAgent() if mode == "ai" else None

    running = True
    while running:
        action = ACTION_NONE
        vertical_input = 0  # -1 up, +1 down, 0 none

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if mode == "human":
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action = ACTION_LEFT
            elif keys[pygame.K_RIGHT]:
                action = ACTION_RIGHT
            elif keys[pygame.K_UP]:
                action = ACTION_UP
                vertical_input = -1
            elif keys[pygame.K_DOWN]:
                action = ACTION_DOWN
                vertical_input = 1
        else:
            action = agent.act(state)

        state, reward, terminated, truncated, info = env.step(action, vertical_input)

        if terminated or truncated:
            if terminated:
                print(f"Completed level {level_index + 1}")
                level_index += 1
            else:
                print(f"Level {level_index + 1} timed out")

            if level_index >= len(LEVELS):
                print("All levels completed!")
                running = False
                continue

            env.close()
            env = PlatformerEnv(
                LEVELS[level_index],
                render=True,
                run_label=mode,
                level_name=f"level_{level_index + 1}",
            )
            state, info = env.reset()

        clock.tick(10)

    pygame.quit()


if __name__ == "__main__":
    selected_mode = sys.argv[1] if len(sys.argv) > 1 else MODE
    run(selected_mode)
    
