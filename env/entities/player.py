import pygame
from .game_object import GameObject, TILE_SIZE


class Player(GameObject):
    def __init__(self, x, y):
        super().__init__(x, y, (50, 100, 255))
        self.vx = 0.0

    def update(self, action, walls):
        acceleration = 0.2
        friction = 0.85
        max_speed = 0.6

        if action == 0:
            self.vx -= acceleration
        elif action == 1:
            self.vx += acceleration

        self.vx = max(-max_speed, min(max_speed, self.vx))
        self.vx *= friction

        new_x = self.x + self.vx

        future_rect = pygame.Rect(
            int(new_x * TILE_SIZE),
            self.y * TILE_SIZE,
            TILE_SIZE,
            TILE_SIZE
        )

        collision = False
        for wall in walls:
            if future_rect.colliderect(wall.rect()):
                collision = True
                break

        if not collision:
            self.x = new_x
        else:
            self.vx = 0