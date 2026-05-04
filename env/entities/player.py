import pygame
from .game_object import GameObject, TILE_SIZE


class Player(GameObject):
    def __init__(self, x, y):
        super().__init__(x, y, (50, 100, 255))
        self.vx = 0.0
        self.vy = 0.0

    def update(self, action, walls):
        acceleration = 0.2
        friction = 0.85
        max_speed = 0.6

        if action == 0:
            self.vx -= acceleration
        elif action == 1:
            self.vx += acceleration
        elif action == 2:
            self.vy -= acceleration
        elif action == 3:
            self.vy += acceleration

        self.vx = max(-max_speed, min(max_speed, self.vx))
        self.vy = max(-max_speed, min(max_speed, self.vy))
        self.vx *= friction
        self.vy *= friction

        new_x = self.x + self.vx
        new_y = self.y + self.vy

        future_x_rect = pygame.Rect(
            int(new_x * TILE_SIZE),
            int(self.y * TILE_SIZE),
            TILE_SIZE,
            TILE_SIZE
        )

        future_y_rect = pygame.Rect(
            int(self.x * TILE_SIZE),
            int(new_y * TILE_SIZE),
            TILE_SIZE,
            TILE_SIZE
        )

        x_collision = any(future_x_rect.colliderect(wall.rect()) for wall in walls)
        y_collision = any(future_y_rect.colliderect(wall.rect()) for wall in walls)

        if not x_collision:
            self.x = new_x
        else:
            self.vx = 0

        if not y_collision:
            self.y = new_y
        else:
            self.vy = 0
