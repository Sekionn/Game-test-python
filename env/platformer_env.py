from datetime import datetime
from pathlib import Path
import time
from env.entities import Player, Wall, Door
from env.entities.game_object import TILE_SIZE
import pygame

RESULTS_FILE = Path("game_results.txt")


WALL = 1
EMPTY = 0
DOOR = 3
PLAYER = 9

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
        self.done = False
        self.steps = 0
        self.input_count = 0
        self.has_moved = False
        self.started_at = time.perf_counter()
        self.finished_at = None
        self.previous_distance_to_door = self._distance_to_door()

        self.walls = []
        self.players = []   # changed from single player → list
        self.door = None

        for y, row in enumerate(self.level):
            for x, tile in enumerate(row):
                if tile == WALL:
                    self.walls.append(Wall(x, y))
                elif tile == DOOR:
                    self.door = Door(x, y)
                elif tile == PLAYER:
                    self.players.append(Player(x, y))  # multiple players

        if len(self.players) == 0:
            raise ValueError("No player spawn (9) found in level")

        if self.door is None:
            raise ValueError("No door (3) found in level")

        self.done = False

        return self._get_state()

    def step(self, action):
        reward = STEP_PENALTY
        self.steps += 1

        if action in (0, 1):
            self.has_moved = True
            self.input_count += 1

        # SAME action applied to ALL players
        for player in self.players:
            player.update(action, self.walls)


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

        # check win condition (any player reaches door)
        for player in self.players:
            if player.rect().colliderect(self.door.rect()):
                reward += 10
                self.done = True
                break

        if self.render_mode:
            self._render()

        return self._get_state(), reward, self.done

    def _get_state(self):
        state = []
        for p in self.players:
            state.extend([p.x, p.vx])
        return state

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

        for wall in self.walls:
            wall.render(self.screen)

        self.door.render(self.screen)

        for player in self.players:
            player.render(self.screen)

        if not self.has_moved:
            timer = self.font.render(f"{self._elapsed_seconds():.1f}s", True, (20, 20, 20))
            self.screen.blit(timer, (12, 12))

        pygame.display.flip()
        self.clock.tick(10)
