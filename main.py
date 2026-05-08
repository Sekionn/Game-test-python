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
        [1,1,1,1,1,4,0,0,0,5,1,6,1,6,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    ],
    #Level 4
    [
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,1],
        [1,1,9,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1],
        [1,1,1,1,1,4,0,0,0,0,0,0,5,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,6,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    ],
    #Level 5
    [
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,8,0,0,0,0,0,0,0,0,0,0,3,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,0,0,0,0,0,0,0,0,0,0,0,9,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    ],

]

MODE = "human"
PlayerName = "Sebastian"

def run(mode):
    level_index = 0
    env = PlatformerEnv(
        LEVELS[level_index],
        render=True,
        run_label=mode,
        level_name=f"level_{level_index + 1}",
        player_name=PlayerName
    )
    state, info = env.reset()

    clock = pygame.time.Clock()
    agent = RandomAgent() if mode == "ai" else None

    running = True
    while running:
        action = 2

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
                action = 3
            elif keys[pygame.K_DOWN]:
                action = 4
            elif keys[pygame.K_r]:
                action = 5
        else:
            action = agent.act(state)

        if  action == 5:
            env.softReset()
        
        state, reward, terminated, truncated, info = env.step(action)

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
                player_name=PlayerName
            )
            state, info = env.reset()

        clock.tick(10)

    pygame.quit()


if __name__ == "__main__":
    selected_mode = sys.argv[1] if len(sys.argv) > 1 else MODE
    run(selected_mode)
    
