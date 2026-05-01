import pygame

TILE_SIZE = 60

WALL = 1
EMPTY = 0
BUTTON = 2
DOOR = 3


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
        self.player_x = 1.0
        self.player_vx = 0.0
        self.player_y = 4

        self.button_pressed = False
        self.done = False

        return self._get_state()

    def step(self, action):
        reward = -0.01

        acceleration = 0.2
        friction = 0.5
        max_speed = 0.6

        # --- INPUT → acceleration ---
        if action == 0:      # left
            self.player_vx -= acceleration
        elif action == 1:    # right
            self.player_vx += acceleration

        # --- clamp speed ---
        if self.player_vx > max_speed:
            self.player_vx = max_speed
        if self.player_vx < -max_speed:
            self.player_vx = -max_speed

        # --- friction (natural slowdown) ---
        self.player_vx *= friction

        # --- apply motion ---
        new_x = self.player_x + self.player_vx
        new_y = self.player_y

        # collision (convert float → grid)
        if self.level[new_y][int(new_x)] != 1:
            self.player_x = new_x

        tile = self.level[self.player_y][int(self.player_x)]

        # button logic
        if tile == 2:
            if not self.button_pressed:
                reward += 1
            self.button_pressed = True

        # door logic
        if tile == 3 and self.button_pressed:
            reward += 10
            self.done = True

        if self.render_mode:
            self._render()

        return self._get_state(), reward, self.done

    def _get_state(self):
        return [
            self.player_x,
            self.player_vx,
            self.button_pressed
        ]

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
                    color = (0, 255, 0) if self.button_pressed else (255, 0, 0)
                    pygame.draw.rect(self.screen, color, rect)

        # player
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

        pygame.display.flip()
        self.clock.tick(10)