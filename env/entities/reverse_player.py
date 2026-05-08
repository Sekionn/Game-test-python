from env.entities.extender import Extender
import pygame
from .game_object import GameObject, TILE_SIZE


class ReversePlayer(GameObject):
    def __init__(self, x, y):
        super().__init__(x, y, (182, 80, 180))
        self.vx = 0.0
        self.vy = 0.0

    def update(self, action, walls, extenders: list[Extender], players=None):
        if players is None:
            players = []

        gravity = 0.4
        max_fall = 5

        dx = 0
        if action == 0:
            dx = 1
        elif action == 1:
            dx = -1

        self.vx = dx

        self.vy += gravity
        self.vy = min(self.vy, max_fall)

        solid_rects = self._solid_rects(walls, extenders)

        new_x = self.x + dx
        future_rect_x = pygame.Rect(
            int(new_x * TILE_SIZE),
            int(self.y * TILE_SIZE),
            TILE_SIZE,
            TILE_SIZE
        )

        horizontal_hits = [
            rect for rect in solid_rects
            if future_rect_x.colliderect(rect)
        ]

        if not horizontal_hits:
            self.x = new_x
        else:
            if dx > 0:
                future_rect_x.right = min(rect.left for rect in horizontal_hits)
            elif dx < 0:
                future_rect_x.left = max(rect.right for rect in horizontal_hits)

            self.x = future_rect_x.left / TILE_SIZE

        if horizontal_hits:
            self.vx = 0

        new_y = self.y + self.vy
        future_rect = pygame.Rect(
            int(self.x * TILE_SIZE),
            int(new_y * TILE_SIZE),
            TILE_SIZE,
            TILE_SIZE
        )

        vertical_rects = list(solid_rects)
        for other in players:
            if other is self:
                continue
            vertical_rects.append(other.rect())

        vertical_hits = [
            rect for rect in vertical_rects
            if future_rect.colliderect(rect)
        ]

        if not vertical_hits:
            self.y = new_y
        else:
            current_rect = self.rect()

            if self.vy > 0:
                landing_hits = [
                    rect for rect in vertical_hits
                    if self._is_landing_on_rect(future_rect, rect)
                ]

                if landing_hits:
                    future_rect.bottom = min(rect.top for rect in landing_hits)
                    self.y = future_rect.top / TILE_SIZE
                    self.vy = 0
                else:
                    self.y = new_y

            elif self.vy < 0:
                ceiling_hits = [
                    rect for rect in vertical_hits
                    if current_rect.top >= rect.bottom - 5
                ]

                if ceiling_hits:
                    future_rect.top = max(rect.bottom for rect in ceiling_hits)
                    self.y = future_rect.top / TILE_SIZE
                    self.vy = 0
                else:
                    self.y = new_y

    def _is_landing_on_rect(self, future_rect, ground_rect):
        if self.vy <= 0:
            return False

        current_rect = self.rect()

        horizontal_overlap = (
            future_rect.right > ground_rect.left and
            future_rect.left < ground_rect.right
        )

        was_above_or_touching = current_rect.bottom <= ground_rect.top + 5
        will_touch_or_pass_top = future_rect.bottom >= ground_rect.top

        return horizontal_overlap and was_above_or_touching and will_touch_or_pass_top

    def _solid_rects(self, walls, extenders):
        rects = [wall.rect() for wall in walls]

        for extender in extenders:
            rects.extend(extender.get_rects())

        return rects
