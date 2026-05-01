from .game_object import GameObject


class Door(GameObject):
    def __init__(self, x, y):
        super().__init__(x, y, (0, 200, 0))