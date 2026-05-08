import pygame

from agents.q_learning_agent import QLearningAgent
from env.platformer_env import PlatformerEnv
from main import LEVELS

Q_TABLE_PATH = "q_table.json"


def watch():
    agent = QLearningAgent.load(Q_TABLE_PATH)

    for level_index, level in enumerate(LEVELS):
        level_name = f"level_{level_index + 1}"
        env = PlatformerEnv(
            level,
            render=True,
            run_label="q_learning_watch",
            level_name=level_name,
            log_results=True,
        )
        state, info = env.reset()
        terminated = False
        truncated = False
        running = True

        while running and not terminated and not truncated:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            action = agent.act(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)

        env.close()

        if not running:
            break

        print(
            f"{level_name}: "
            f"{'complete' if terminated else 'timeout'}, "
            f"ticks={info['ticks']}, inputs={info['inputs']}"
        )


if __name__ == "__main__":
    watch()
