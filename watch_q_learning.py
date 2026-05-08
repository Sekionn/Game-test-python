import pygame

from agents.q_learning_agent import QLearningAgent
from env.platformer_env import MAX_EPISODE_TICKS, PlatformerEnv
from main import LEVELS

Q_TABLE_PATH = "best_q_table.json"
WATCH_EPISODES_PER_LEVEL = 3
MAX_WATCH_TICKS = MAX_EPISODE_TICKS
USE_EXPLORATION_WHILE_WATCHING = False


def watch():
    agent = QLearningAgent.load(Q_TABLE_PATH)

    for level_index, level in enumerate(LEVELS):
        level_name = f"level_{level_index + 1}"
        for episode in range(1, WATCH_EPISODES_PER_LEVEL + 1):
            env = PlatformerEnv(
                level,
                render=True,
                run_label="q_learning_watch",
                level_name=level_name,
                log_results=True,
                max_ticks=MAX_WATCH_TICKS,
                episode=episode,
            )
            state, info = env.reset()
            terminated = False
            truncated = False
            running = True

            while running and not terminated and not truncated:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                action = agent.act(state, training=USE_EXPLORATION_WHILE_WATCHING)
                state, reward, terminated, truncated, info = env.step(action)

            env.close()

            if not running:
                return

            print(
                f"{level_name} watch episode {episode}: "
                f"{'complete' if terminated else 'timeout'}, "
                f"ticks={info['ticks']}, "
                f"inputs={info['inputs']}, "
                f"distance={info['distance_to_door']:.2f}"
            )


if __name__ == "__main__":
    watch()
