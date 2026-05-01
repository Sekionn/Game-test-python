from datetime import datetime
from pathlib import Path
import time

import pygame

TILE_SIZE = 60
RESULTS_FILE = Path("game_results.txt")

WALL = 1
EMPTY = 0
BUTTON = 2
DOOR = 3

STEP_PENALTY = -0.01
DISTANCE_REWARD_SCALE = 0.75
COMPLETION_REWARD = 100
TARGET_COMPLETION_TICKS = 56
TARGET_INPUTS = 56
FAST_COMPLETION_REWARD = 25
INPUT_EFFICIENCY_REWARD = 25


class PlatformerEnv:
    def __init__(self, level, render=True, run_label="unknown", level_name="level_1"):
        self.level = level
        self.render_mode = render
        self.run_label = run_label
        self.level_name = level_name

        self.height = len(level)
        self.width = len(level[0])
        self.door_x, self.door_y = self._find_tile(DOOR)
        self.start_x = 1.0
        self.start_y = self.door_y

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.width * TILE_SIZE, self.height * TILE_SIZE)
            )
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 28)

        self.reset()

    def reset(self):
        self.player_x = self.start_x
        self.player_vx = 0.0
        self.player_y = self.start_y

        self.done = False
        self.steps = 0
        self.input_count = 0
        self.has_moved = False
        self.started_at = time.perf_counter()
        self.finished_at = None
        self.previous_distance_to_door = self._distance_to_door()

        return self._get_state()

    def step(self, action):
        reward = STEP_PENALTY
        self.steps += 1

        if action in (0, 1):
            self.has_moved = True
            self.input_count += 1

        acceleration = 0.2
        friction = 0.5
        max_speed = 0.6

        if action == 0:
            self.player_vx -= acceleration
        elif action == 1:
            self.player_vx += acceleration

        if self.player_vx > max_speed:
            self.player_vx = max_speed
        if self.player_vx < -max_speed:
            self.player_vx = -max_speed

        self.player_vx *= friction

        new_x = self.player_x + self.player_vx
        new_y = self.player_y

        if self.level[new_y][int(new_x)] != WALL:
            self.player_x = new_x

        current_distance = self._distance_to_door()
        distance_delta = self.previous_distance_to_door - current_distance
        reward += distance_delta * DISTANCE_REWARD_SCALE
        self.previous_distance_to_door = current_distance

        tile = self.level[self.player_y][int(self.player_x)]

        if tile == DOOR:
            elapsed_seconds = self._elapsed_seconds()
            speed_reward = self._completion_efficiency_reward(
                self.steps,
                TARGET_COMPLETION_TICKS,
                FAST_COMPLETION_REWARD
            )
            input_reward = self._completion_efficiency_reward(
                self.input_count,
                TARGET_INPUTS,
                INPUT_EFFICIENCY_REWARD
            )
            reward += COMPLETION_REWARD + speed_reward + input_reward
            self.done = True
            self.finished_at = time.perf_counter()
            self._record_completion(elapsed_seconds, reward)

        if self.render_mode:
            self._render()

        return self._get_state(), reward, self.done

    def _get_state(self):
        return [
            self.player_x,
            self.player_vx,
        ]

    def _find_tile(self, target_tile):
        for y, row in enumerate(self.level):
            for x, tile in enumerate(row):
                if tile == target_tile:
                    return x, y

        raise ValueError(f"Level does not contain tile {target_tile}.")

    def _distance_to_door(self):
        return abs(self.door_x - self.player_x) + abs(self.door_y - self.player_y)

    def _elapsed_seconds(self):
        return time.perf_counter() - self.started_at

    def _completion_efficiency_reward(self, actual_value, target_value, max_reward):
        if actual_value <= target_value:
            return max_reward

        return max(0, max_reward * (target_value / actual_value))

    def _record_completion(self, elapsed_seconds, final_reward):
        header = (
            "timestamp,run_label,level,ticks,inputs,elapsed_seconds,final_reward\n"
        )
        row = (
            f"{datetime.now().isoformat(timespec='seconds')},"
            f"{self.run_label},"
            f"{self.level_name},"
            f"{self.steps},"
            f"{self.input_count},"
            f"{elapsed_seconds:.3f},"
            f"{final_reward:.3f}\n"
        )

        if not RESULTS_FILE.exists():
            RESULTS_FILE.write_text(header, encoding="utf-8")

        with RESULTS_FILE.open("a", encoding="utf-8") as results_file:
            results_file.write(row)

    def _render(self):
        self.screen.fill((255, 255, 255))

        for y in range(self.height):
            for x in range(self.width):
                tile = self.level[y][x]

                rect = pygame.Rect(
                    x * TILE_SIZE,
                    y * TILE_SIZE,
                    TILE_SIZE,
                    TILE_SIZE
                )

                if tile == WALL:
                    pygame.draw.rect(self.screen, (80, 80, 80), rect)
                elif tile == BUTTON:
                    pygame.draw.rect(self.screen, (200, 50, 50), rect)
                elif tile == DOOR:
                    pygame.draw.rect(self.screen, (0, 255, 0), rect)

        pygame.draw.rect(
            self.screen,
            (50, 100, 255),
            pygame.Rect(
                int(self.player_x * TILE_SIZE),
                self.player_y * TILE_SIZE,
                TILE_SIZE,
                TILE_SIZE
            )
        )

        if not self.has_moved:
            timer = self.font.render(f"{self._elapsed_seconds():.1f}s", True, (20, 20, 20))
            self.screen.blit(timer, (12, 12))

        pygame.display.flip()
        self.clock.tick(10)
