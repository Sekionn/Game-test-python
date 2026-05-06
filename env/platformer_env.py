from datetime import datetime
from pathlib import Path
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from env.entities import Player, Wall, Door, Extender, ReversePlayer
from env.entities.game_object import TILE_SIZE

RESULTS_FILE = Path("game_results.txt")
RESULTS_HEADER = (
    "timestamp,run_label,level,outcome,ticks,inputs,elapsed_seconds,final_reward\n"
)

WALL = 1
DOOR = 3
EXTENDER_X_RIGHT = 4
EXTENDER_X_LEFT = 5

EXTENDER_Y = 6
REVERSEPLAYER = 8
PLAYER = 9


STEP_PENALTY = -0.01
DISTANCE_REWARD_SCALE = 0.75
COMPLETION_REWARD = 100
TARGET_COMPLETION_TICKS = 56
TARGET_INPUTS = 56
FAST_COMPLETION_REWARD = 25
INPUT_EFFICIENCY_REWARD = 25
MAX_EPISODE_TICKS = 300


class PlatformerEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        level,
        render=True,
        run_label="unknown",
        level_name="level_1",
        log_results=True,
        max_ticks=MAX_EPISODE_TICKS,
    ):
        super().__init__()
        self.level = level
        self.render_mode = "human" if render else None
        self.run_label = run_label
        self.level_name = level_name
        self.log_results = log_results
        self.max_ticks = max_ticks

        self.height = len(level)
        self.width = len(level[0])
        self.door_x, self.door_y = self._find_tile(DOOR)
        self.player_spawn_count = self._count_tile(PLAYER)

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -0.6, -0.6] * self.player_spawn_count, dtype=np.float32),
            high=np.array(
                [float(self.width), float(self.height), 0.6, 0.6] * self.player_spawn_count,
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.width * TILE_SIZE, self.height * TILE_SIZE)
            )
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 28)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.done = False
        self.steps = 0
        self.input_count = 0
        self.has_moved = False
        self.started_at = time.perf_counter()
        self.finished_at = None
        self.episode_recorded = False
        self.walls = []
        self.players = []
        self.extenders = []
        self.extender_groups = {
            "x": [],
            "y": []
        }

        self.door = None

        for y, row in enumerate(self.level):
            for x, tile in enumerate(row):
                if tile == WALL:
                    self.walls.append(Wall(x, y))
                elif tile == DOOR:
                    self.door = Door(x, y)
                elif tile == PLAYER:
                    self.players.append(Player(x, y))
                elif tile == EXTENDER_X_RIGHT:
                    e = Extender(x, y, "x", 1)
                    self.extenders.append(e)
                    self.extender_groups["x"].append(e)
                elif tile == EXTENDER_X_LEFT:
                    e = Extender(x, y, "x", -1)
                    self.extenders.append(e)
                    self.extender_groups["x"].append(e)
                elif tile == EXTENDER_Y:  
                    e = Extender(x, y, "y", -1)
                    self.extenders.append(e)
                    self.extender_groups["y"].append(e)
                elif tile == REVERSEPLAYER:
                    self.players.append(ReversePlayer(x, y))

        if len(self.players) == 0:
            raise ValueError("No player spawn (9) found in level.")

        if self.door is None:
            raise ValueError("No door (3) found in level.")

        self.previous_distance_to_door = self._distance_to_door()

        if self.render_mode == "human":
            self._render()

        return self._get_obs(), self._get_info()

    def step(self, action, vertical_input=2):
        reward = STEP_PENALTY
        self.steps += 1

        if action in (0, 1, 2, 3):
            self.has_moved = True
            self.input_count += 1

        group_action = {
            "x": 0,  # -1 retract, 1 extend
            "y": 0
        }

        if action in (0, 1):
            # X axis controls
            if action == 1:
                group_action["x"] = 1
            elif action == 0:
                group_action["x"] = -1

        elif action in (3, 4):
            # Y axis controls
            if action == 3:
                group_action["y"] = 1
            elif action == 4:
                group_action["y"] = -1

        extenders_by_action = sorted(
            self.extenders,
            key=lambda extender: extender.action_for_group_action(group_action)
        )

        # update retracting extenders before extending extenders
        for extender in extenders_by_action:
            extender.update(
                group_action,
                self.players,
                self.walls,
                self.door,
                self.extender_groups[extender.axis]
            )

        for player in sorted(self.players, key=lambda player: player.y, reverse=True):
            player.update(
                action,
                self.walls,
                self.extenders,
                self.players
            )

        current_distance = self._distance_to_door()
        distance_delta = self.previous_distance_to_door - current_distance
        distance_reward = distance_delta * DISTANCE_REWARD_SCALE
        reward += distance_reward
        self.previous_distance_to_door = current_distance

        terminated = any(player.rect().colliderect(self.door.rect()) for player in self.players)
        truncated = self.steps >= self.max_ticks

        completion_reward = 0
        speed_reward = 0
        input_reward = 0

        if terminated:
            elapsed_seconds = self._elapsed_seconds()
            speed_reward = self._completion_efficiency_reward(
                self.steps,
                TARGET_COMPLETION_TICKS,
                FAST_COMPLETION_REWARD,
            )
            input_reward = self._completion_efficiency_reward(
                self.input_count,
                TARGET_INPUTS,
                INPUT_EFFICIENCY_REWARD,
            )
            completion_reward = COMPLETION_REWARD
            reward += completion_reward + speed_reward + input_reward
            self.done = True
            self.finished_at = time.perf_counter()

        if terminated or truncated:
            self.done = True
            self.finished_at = time.perf_counter()

            if self.log_results and not self.episode_recorded:
                outcome = "complete" if terminated else "timeout"
                self._record_episode(outcome, self._elapsed_seconds(), reward)
                self.episode_recorded = True

        if self.render_mode:
            self._render()

        info = self._get_info()
        info.update(
            {
                "distance_reward": distance_reward,
                "completion_reward": completion_reward,
                "speed_reward": speed_reward,
                "input_reward": input_reward,
            }
        )

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        state = []
        for player in self.players:
            state.extend([player.x, player.y, player.vx, player.vy])

        return np.array(state, dtype=np.float32)

    def _get_info(self):
        return {
            "distance_to_door": self._distance_to_door(),
            "ticks": self.steps,
            "inputs": self.input_count,
            "level": self.level_name,
        }

    def _find_tile(self, target_tile):
        for y, row in enumerate(self.level):
            for x, tile in enumerate(row):
                if tile == target_tile:
                    return x, y

        raise ValueError(f"Level does not contain tile {target_tile}.")

    def _count_tile(self, target_tile):
        count = 0
        for row in self.level:
            count += row.count(target_tile)
        return count

    def _distance_to_door(self):
        distances = [
            abs(self.door.x - player.x) + abs(self.door.y - player.y)
            for player in self.players
        ]
        return min(distances)

    def _elapsed_seconds(self):
        return time.perf_counter() - self.started_at

    def _completion_efficiency_reward(self, actual_value, target_value, max_reward):
        if actual_value <= target_value:
            return max_reward

        return max(0, max_reward * (target_value / actual_value))

    def _record_episode(self, outcome, elapsed_seconds, final_reward):
        row = (
            f"{datetime.now().isoformat(timespec='seconds')},"
            f"{self.run_label},"
            f"{self.level_name},"
            f"{outcome},"
            f"{self.steps},"
            f"{self.input_count},"
            f"{elapsed_seconds:.3f},"
            f"{final_reward:.3f}\n"
        )

        if not RESULTS_FILE.exists():
            RESULTS_FILE.write_text(RESULTS_HEADER, encoding="utf-8")

        with RESULTS_FILE.open("a", encoding="utf-8") as results_file:
            results_file.write(row)

    def _render(self):
        self.screen.fill((255, 255, 255))

        for extender in self.extenders:
            extender.render(self.screen)

        for wall in self.walls:
            wall.render(self.screen)

        self.door.render(self.screen)

        for player in self.players:
            player.render(self.screen)

        if not self.has_moved:
            timer = self.font.render(f"{self._elapsed_seconds():.1f}s", True, (20, 20, 20))
            self.screen.blit(timer, (12, 12))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def get_active_extenders(self):
        active = []
        for e in self.extenders:
            if len(e.get_players_on_top(self.players)) > 0:
                active.append(e)
        return active
