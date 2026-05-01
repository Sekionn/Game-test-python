import pygame
from env.platformer_env import PlatformerEnv
from agents.random_agent import RandomAgent

LEVEL = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,2,0,0,3,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
]

MODE = "human"

def run(mode):
    env = PlatformerEnv(LEVEL, render=True)
    state = env.reset()
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
                action = 0
            elif keys[pygame.K_RIGHT]:
                action = 1
        else:
            action = agent.act(state)

        state, reward, done = env.step(action)

        if done:
            state = env.reset()

        clock.tick(10)

    pygame.quit()

if __name__ == "__main__":
    run(MODE)
    