from .game_object import GameObject


class Wall(GameObject):
    def __init__(self, x, y):
        super().__init__(x, y, (80, 80, 80))