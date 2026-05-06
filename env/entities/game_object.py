import pygame

TILE_SIZE = 60


class GameObject:
    def __init__(self, x, y, color):
        self.x = float(x)
        self.y = y
        self.color = color

    def rect(self):
        return pygame.Rect(
            int(self.x * TILE_SIZE),
            int(self.y * TILE_SIZE),
            TILE_SIZE,
            TILE_SIZE
        )

    def render(self, screen):
        pygame.draw.rect(screen, self.color, self.rect())
