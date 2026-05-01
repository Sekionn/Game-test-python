import pygame
from env.entities import Player, Wall, Door
from env.entities.game_object import TILE_SIZE

WALL = 1
EMPTY = 0
DOOR = 3
PLAYER = 9


class PlatformerEnv:
    def __init__(self, level, render=True):
        self.level = level
        self.render_mode = render

        self.height = len(level)
        self.width = len(level[0])

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.width * TILE_SIZE, self.height * TILE_SIZE)
            )
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.walls = []
        self.players = []
        self.door = None

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
        return self._get_state()

    def step(self, action):
        reward = -0.01

        for player in self.players:
            player.update(action, self.walls)

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

    def _render(self):
        self.screen.fill((255, 255, 255))

        for wall in self.walls:
            wall.render(self.screen)

        self.door.render(self.screen)

        for player in self.players:
            player.render(self.screen)

        pygame.display.flip()
        self.clock.tick(10)