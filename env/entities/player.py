import pygame
from .game_object import GameObject, TILE_SIZE


class Player(GameObject):
    def __init__(self, x, y):
        super().__init__(x, y, (50, 100, 255))
        self.vx = 0.0
        self.vy = 0.0

    def update(self, action, walls, extenders):
        # --- PHYSICS CONSTANTS ---
        acceleration = 25
        friction = 0.01
        max_speed = 25
        gravity = 0.4
        max_fall = 5

        # --- INPUT (horizontal only) ---
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

        # --- GRAVITY ---
        self.vy += gravity
        self.vy = min(self.vy, max_fall)

        # --- HORIZONTAL MOVE ---
        new_x = self.x + self.vx
        new_y = self.y + self.vy

        future_rect_x = pygame.Rect(
            int(new_x * TILE_SIZE),
            int(self.y * TILE_SIZE),
            TILE_SIZE,
            TILE_SIZE
        )

        collision = False
        for wall in walls:
            if future_rect_x.colliderect(wall.rect()):
                collision = True
                break

        x_collision = any(future_x_rect.colliderect(wall.rect()) for wall in walls)
        y_collision = any(future_y_rect.colliderect(wall.rect()) for wall in walls)

        if not x_collision:
            self.x = new_x
        else:
            self.vx = 0

        # --- VERTICAL MOVE ---
        new_y = self.y + self.vy

        future_rect = pygame.Rect(
            int(self.x * TILE_SIZE),
            int(new_y * TILE_SIZE),
            TILE_SIZE,
            TILE_SIZE
        )

        # assume falling
        grounded = False
        lowest_ground_y = None

        # --- check walls ---
        for wall in walls:
            wall_rect = wall.rect()

            if future_rect.colliderect(wall_rect):
                if self.vy > 0:  # falling
                    lowest_ground_y = min(lowest_ground_y or wall.y, wall.y)
                    grounded = True

        # --- check extenders (IMPORTANT FIX) ---
        for extender in extenders:
            for rect in extender.get_rects():

                # ONLY check landing from above
                if self.vy > 0:
                    # proper overlap check (not exact equality)
                    if (
                        self.x * TILE_SIZE + TILE_SIZE > rect.left and
                        self.x * TILE_SIZE < rect.right and
                        future_rect.bottom >= rect.top and
                        self.rect().bottom <= rect.top + 10
                    ):
                        lowest_ground_y = min(lowest_ground_y or rect.top // TILE_SIZE, rect.top // TILE_SIZE)
                        grounded = True

        # --- resolve landing ---
        if grounded:
            self.vy = 0
            self.y = lowest_ground_y - 1 if lowest_ground_y is not None else self.y
        else:
            self.y = new_y

        # --- PLATFORM CARRY (NEW STEP) ---
        player_rect = self.rect()

        for extender in extenders:
            for rect in extender.get_rects():

                is_standing_on = (
                    self.vy >= 0 and
                    abs(player_rect.bottom - rect.top) <= 5 and
                    player_rect.right > rect.left and
                    player_rect.left < rect.right
                )

                if is_standing_on:
                    self.x += extender.delta_x
                    self.y += extender.delta_y
