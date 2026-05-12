import math

import pygame

from agents.q_learning_agent import QLearningAgent
from env.entities.game_object import TILE_SIZE
from env.platformer_env import MAX_EPISODE_TICKS, PlatformerEnv
from main import LEVELS

LEVEL_INDEX = 0
GENERATIONS = 5
ATTEMPTS_PER_GENERATION = 100
VISIBLE_ATTEMPTS = 10
SIMULATION_STEPS_PER_FRAME = 1
Q_TABLE_PATH = "q_table.json"
BEST_Q_TABLE_PATH = "best_q_table.json"
EVALUATION_ATTEMPTS = 5
EVENT_CHECK_INTERVAL = 5

MINI_TILE_SIZE = 16
GRID_COLUMNS = 5
PANEL_GAP = 8
HEADER_HEIGHT = 44
BACKGROUND = (24, 24, 28)
PANEL_BACKGROUND = (245, 245, 245)
WALL_COLOR = (80, 80, 80)
DOOR_COLOR = (0, 200, 0)
PLAYER_COLOR = (50, 100, 255)
EXTENDER_COLOR = (200, 100, 50)
TEXT_COLOR = (230, 230, 230)
PANEL_TEXT_COLOR = (20, 20, 20)


def visual_train():
    """Train Q-learning while drawing several attempts at the same time.

    The visible attempts all update the same Q-table. When one attempt finishes,
    it is replaced with the next attempt in the current generation until all
    attempts for that generation are used.
    """
    pygame.init()
    font = pygame.font.SysFont(None, 22)

    level = LEVELS[LEVEL_INDEX]
    level_name = f"level_{LEVEL_INDEX + 1}"
    level_width = len(level[0])
    level_height = len(level)
    panel_width = level_width * MINI_TILE_SIZE
    panel_height = level_height * MINI_TILE_SIZE + 22
    grid_rows = math.ceil(VISIBLE_ATTEMPTS / GRID_COLUMNS)
    screen_width = GRID_COLUMNS * panel_width + (GRID_COLUMNS + 1) * PANEL_GAP
    screen_height = HEADER_HEIGHT + grid_rows * panel_height + (grid_rows + 1) * PANEL_GAP
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Q-learning visual training")
    clock = pygame.time.Clock()

    agent = QLearningAgent()
    best_agent = None
    best_score = None
    running = True

    for generation in range(1, GENERATIONS + 1):
        if not running:
            break

        envs = []
        states = []
        rewards = []
        active_attempts = []
        completed = 0
        next_attempt = 1

        for _ in range(min(VISIBLE_ATTEMPTS, ATTEMPTS_PER_GENERATION)):
            env, state = _new_env(level, level_name, generation, next_attempt)
            envs.append(env)
            states.append(state)
            rewards.append(0)
            active_attempts.append(next_attempt)
            next_attempt += 1

        finished_attempts = 0

        while running and finished_attempts < ATTEMPTS_PER_GENERATION:
            running = _handle_events()

            for _ in range(SIMULATION_STEPS_PER_FRAME):
                for index, env in enumerate(envs):
                    if not running:
                        break

                    if index % EVENT_CHECK_INTERVAL == 0:
                        running = _handle_events()
                        if not running:
                            break

                    if env is None:
                        continue

                    action = agent.act(states[index], training=True)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    agent.learn(states[index], action, reward, next_state, terminated or truncated)
                    states[index] = next_state
                    rewards[index] += reward

                    if terminated or truncated:
                        if terminated:
                            completed += 1

                        finished_attempts += 1
                        agent.finish_episode()
                        env.close()

                        if next_attempt <= ATTEMPTS_PER_GENERATION:
                            env, state = _new_env(level, level_name, generation, next_attempt)
                            envs[index] = env
                            states[index] = state
                            rewards[index] = 0
                            active_attempts[index] = next_attempt
                            next_attempt += 1
                        else:
                            envs[index] = None

                if not running:
                    break

            _draw(
                screen,
                font,
                envs,
                active_attempts,
                generation,
                finished_attempts,
                completed,
                agent.epsilon,
                panel_width,
                panel_height,
            )
            pygame.display.flip()
            clock.tick(30)

        for env in envs:
            if env is not None:
                env.close()

        evaluation = _evaluate_agent(agent, level, level_name, generation)
        print(
            f"{level_name} generation {generation:02d} evaluation: "
            f"{evaluation['completed']}/{EVALUATION_ATTEMPTS} complete, "
            f"avg ticks={evaluation['average_ticks']:.1f}, "
            f"score={evaluation['score']:.2f}"
        )

        if best_score is None or evaluation["score"] > best_score:
            best_score = evaluation["score"]
            best_agent = agent.clone()
            best_agent.save(BEST_Q_TABLE_PATH)
            print(f"New best Q-table saved from generation {generation:02d}")

    agent.save(Q_TABLE_PATH)
    if best_agent is not None:
        best_agent.save(BEST_Q_TABLE_PATH)
    pygame.quit()
    print(f"Saved learned Q-table to {Q_TABLE_PATH}")
    print(f"Saved best Q-table to {BEST_Q_TABLE_PATH}")


def _new_env(level, level_name, generation, attempt):
    env = PlatformerEnv(
        level,
        render=False,
        run_label="q_learning_visual_train",
        level_name=level_name,
        log_results=True,
        max_ticks=MAX_EPISODE_TICKS,
        generation=generation,
        episode=attempt,
    )
    state, info = env.reset()
    return env, state


def _handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False

    return True


def _evaluate_agent(agent, level, level_name, generation):
    completed = 0
    total_ticks = 0
    total_reward = 0

    for attempt in range(1, EVALUATION_ATTEMPTS + 1):
        env = PlatformerEnv(
            level,
            render=False,
            run_label="q_learning_visual_evaluation",
            level_name=level_name,
            log_results=True,
            max_ticks=MAX_EPISODE_TICKS,
            generation=generation,
            episode=attempt,
        )
        state, info = env.reset()
        terminated = False
        truncated = False
        attempt_reward = 0

        while not terminated and not truncated:
            action = agent.act(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            attempt_reward += reward

        if terminated:
            completed += 1

        total_ticks += info["ticks"]
        total_reward += attempt_reward
        env.close()

    average_ticks = total_ticks / EVALUATION_ATTEMPTS
    average_reward = total_reward / EVALUATION_ATTEMPTS
    score = completed * 10000 - average_ticks + average_reward

    return {
        "completed": completed,
        "average_ticks": average_ticks,
        "average_reward": average_reward,
        "score": score,
    }


def _draw(
    screen,
    font,
    envs,
    active_attempts,
    generation,
    finished_attempts,
    completed,
    epsilon,
    panel_width,
    panel_height,
):
    screen.fill(BACKGROUND)
    header = (
        f"Generation {generation} | "
        f"finished {finished_attempts}/{ATTEMPTS_PER_GENERATION} | "
        f"completed {completed} | epsilon {epsilon:.3f}"
    )
    screen.blit(font.render(header, True, TEXT_COLOR), (PANEL_GAP, 14))

    for index, env in enumerate(envs):
        col = index % GRID_COLUMNS
        row = index // GRID_COLUMNS
        x_offset = PANEL_GAP + col * (panel_width + PANEL_GAP)
        y_offset = HEADER_HEIGHT + PANEL_GAP + row * (panel_height + PANEL_GAP)

        pygame.draw.rect(
            screen,
            PANEL_BACKGROUND,
            pygame.Rect(x_offset, y_offset, panel_width, panel_height),
        )

        if env is None:
            label = font.render("done", True, PANEL_TEXT_COLOR)
            screen.blit(label, (x_offset + 6, y_offset + 4))
            continue

        _draw_env(screen, env, x_offset, y_offset + 20)
        info = env._get_info()
        label = (
            f"#{active_attempts[index]} "
            f"t{info['ticks']} "
            f"d{info['distance_to_door']:.1f}"
        )
        screen.blit(font.render(label, True, PANEL_TEXT_COLOR), (x_offset + 4, y_offset + 3))


def _draw_env(screen, env, x_offset, y_offset):
    for wall in env.walls:
        _draw_tile(screen, x_offset, y_offset, wall.x, wall.y, WALL_COLOR)

    for extender in env.extenders:
        for rect in extender.get_rects():
            x = rect.x / TILE_SIZE
            y = rect.y / TILE_SIZE
            _draw_tile(screen, x_offset, y_offset, x, y, EXTENDER_COLOR)

    _draw_tile(screen, x_offset, y_offset, env.door.x, env.door.y, DOOR_COLOR)

    for player in env.players:
        _draw_tile(screen, x_offset, y_offset, player.x, player.y, PLAYER_COLOR)


def _draw_tile(screen, x_offset, y_offset, grid_x, grid_y, color):
    rect = pygame.Rect(
        int(x_offset + grid_x * MINI_TILE_SIZE),
        int(y_offset + grid_y * MINI_TILE_SIZE),
        MINI_TILE_SIZE,
        MINI_TILE_SIZE,
    )
    pygame.draw.rect(screen, color, rect)


if __name__ == "__main__":
    visual_train()
