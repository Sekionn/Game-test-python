import pygame
from env.entities import Player, Wall, Door
from env.entities.game_object import TILE_SIZE
from gymnasium import spaces
import gymnasium as gym

WALL = 1
EMPTY = 0
DOOR = 3
PLAYER = 9


class PlatformerEnv(gym.Env):
    def __init__(self, level, render=None):
        super().__init__()

        self.max_steps = 2000
        self.level = level
        self.render_mode = render

        self.height = len(level)
        self.width = len(level[0])

        # Gym spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(3)

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.width * TILE_SIZE, self.height * TILE_SIZE)
            )
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)

        self.steps = 0

        self.walls = []
        self.players = []
        self.door = None
        self.steps = 0

        for y, row in enumerate(self.level):
            for x, tile in enumerate(row):
                if tile == WALL:
                    self.walls.append(Wall(x, y))
                elif tile == DOOR:
                    self.door = Door(x, y)
                elif tile == PLAYER:
                    self.players.append(Player(x, y))

        if len(self.players) == 0:
            raise ValueError("No player spawn (9) found in level")

        if self.door is None:
            raise ValueError("No door (3) found in level")

        self.done = False
        return self._get_state(), {}

    def step(self, action):
        reward = 0

        player = self.players[0]

        prev_dist = abs(player.x - self.door.x)

        # apply action
        for player in self.players:
            player.update(action, self.walls)

        player = self.players[0]
        new_dist = abs(player.x - self.door.x)

        # reward for getting closer
        reward = (prev_dist - new_dist) * 2.0

        if new_dist > prev_dist:
            reward -=0.1

        # small penalty each step (prevents standing still)
        reward -= 0.02

        terminated = False
        truncated = False

        # check win condition (any player reaches door)
        for player in self.players:
            if player.rect().colliderect(self.door.rect()):
                reward += 10
                self.done = True
                terminated = True
                break

        if self.render_mode:
            self._render()
        
        self.steps += 1

        if self.steps >= self.max_steps:
            truncated = True

        return self._get_state(), reward, terminated, truncated, {}

    def _get_state(self):
        p = self.players[0]
        dx = self.door.x - p.x
        
        dx_bin = round(dx, 1)
        vx_bin = round(p.vx, 2)

        return (dx_bin, vx_bin)

    def _render(self):
        self.screen.fill((255, 255, 255))

        for wall in self.walls:
            wall.render(self.screen)

        self.door.render(self.screen)

        for player in self.players:
            player.render(self.screen)

        pygame.display.flip()
        self.clock.tick(10)