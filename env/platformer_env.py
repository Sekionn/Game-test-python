from datetime import datetime
from pathlib import Path
import time
import csv
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from env.entities import Player, Wall, Door, Extender, ReversePlayer
from env.entities.game_object import TILE_SIZE

RESULTS_FILE = Path("game_results.csv")


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
ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_NONE = 2
ACTION_UP = 3
ACTION_DOWN = 4
ACTION_RESET = 5
RESET_PENALTY = -2.0
NO_PROGRESS_TICK_LIMIT = 20
NO_PROGRESS_PENALTY = -0.05


class PlatformerEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        level,
        render=True,
        run_label="unknown",
        level_name="level_1",
        player_name="ai",
        log_results=True,
        max_ticks=MAX_EPISODE_TICKS,
        generation=None,
        episode=None,
    ):
        super().__init__()
        self.level = level
        self.render_mode = "human" if render else None
        self.run_label = run_label
        self.level_name = level_name
        self.player_name = player_name
        self.log_results = log_results
        self.max_ticks = max_ticks
        self.generation = generation
        self.episode = episode

        self.height = len(level)
        self.width = len(level[0])
        self.door_x, self.door_y = self._find_tile(DOOR)
        self.player_spawn_count = self._count_tile(PLAYER) + self._count_tile(REVERSEPLAYER)

        self.action_space = spaces.Discrete(6)
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
        self.resets = 0
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
        self.best_distance_to_door = self.previous_distance_to_door
        self.ticks_since_progress = 0

        if self.render_mode == "human":
            self._render()

        return self._get_obs(), self._get_info()
    
    def softReset(self, seed=None, options=None):
        self.resets += 1
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
        self.best_distance_to_door = self.previous_distance_to_door
        self.ticks_since_progress = 0

        if self.render_mode == "human":
            self._render()

        return self._get_obs(), self._get_info()

    def step(self, action):
        reward = STEP_PENALTY
        self.steps += 1

        if action == ACTION_RESET:
            self.softReset()
            reward += RESET_PENALTY
            truncated = self.steps >= self.max_ticks

            if truncated:
                self.done = True
                self.finished_at = time.perf_counter()

                if self.log_results and not self.episode_recorded:
                    self._record_episode("timeout", self._elapsed_seconds(), reward)
                    self.episode_recorded = True

            if self.render_mode:
                self._render()

            info = self._get_info()
            info.update(
                {
                    "distance_reward": 0,
                    "completion_reward": 0,
                    "speed_reward": 0,
                    "input_reward": 0,
                    "reset_penalty": RESET_PENALTY,
                }
            )
            return self._get_obs(), reward, False, truncated, info

        if action in (ACTION_LEFT, ACTION_RIGHT, ACTION_UP, ACTION_DOWN):
            self.has_moved = True
            self.input_count += 1

        group_action = {
            "x": 0,  # -1 retract, 1 extend
            "y": 0
        }

        if action in (ACTION_LEFT, ACTION_RIGHT):
            # X axis controls
            if action == ACTION_RIGHT:
                group_action["x"] = 1
            elif action == ACTION_LEFT:
                group_action["x"] = -1

        elif action in (ACTION_UP, ACTION_DOWN):
            # Y axis controls
            if action == ACTION_UP:
                group_action["y"] = 1
            elif action == ACTION_DOWN:
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
                self.extenders
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

        if current_distance < self.best_distance_to_door:
            self.best_distance_to_door = current_distance
            self.ticks_since_progress = 0
        else:
            self.ticks_since_progress += 1

        if self.ticks_since_progress >= NO_PROGRESS_TICK_LIMIT:
            reward += NO_PROGRESS_PENALTY

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
                    "no_progress_penalty": NO_PROGRESS_PENALTY
                    if self.ticks_since_progress >= NO_PROGRESS_TICK_LIMIT
                    else 0,
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
            "resets": self.resets,
            "ticks_since_progress": self.ticks_since_progress,
            "level": self.level_name,
            "generation": self.generation,
            "episode": self.episode,
        }

    def set_episode_context(self, generation=None, episode=None):
        self.generation = generation
        self.episode = episode

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
        data = []

        if not RESULTS_FILE.exists():
            data = [
                [
                    'timestamp',
                    'run_label',
                    'level',
                    'generation',
                    'episode',
                    'player_name',
                    'outcome',
                    'ticks',
                    'inputs',
                    'resets',
                    'elapsed_seconds',
                    'final_reward'
                ],
        ]

        if self.player_name == "":
            self.player_name = "NO NAME!!!!"

        data.append(
            [
                datetime.now().isoformat(timespec='seconds'),
                self.run_label,
                self.level_name,
                self.generation if self.generation is not None else '',
                self.episode if self.episode is not None else '',
                self.player_name,
                outcome,
                self.steps,
                self.input_count,
                self.resets,
                f"{elapsed_seconds:.3f}",
                f"{final_reward:.3f}"
            ]
        )

        with open(RESULTS_FILE, mode='a', newline='', encoding='utf-8') as file:
            # Create a csv.writer object
            writer = csv.writer(file, delimiter=";")
            # Write data to the CSV file
            writer.writerows(data)


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

    def close(self):
        if self.render_mode == "human":
            pygame.quit()
