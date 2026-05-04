import pygame
from .game_object import GameObject, TILE_SIZE


class Extender(GameObject):
    def __init__(self, x, y, axis, direction):
        super().__init__(x, y, (200, 100, 50))
        self.delta_x = 0
        self.delta_y = 0
        self.prev_x = x
        self.prev_y = y
        self.axis = axis              # "x" or "y"
        self.direction = direction    # -1 or 1
        self.length = 1
        self.min_length = 1

    def get_rects(self):
        rects = []

        for i in range(self.length):
            if self.axis == "x":
                rects.append(
                    pygame.Rect(
                        int((self.x + i * self.direction) * TILE_SIZE),
                        self.y * TILE_SIZE,
                        TILE_SIZE,
                        TILE_SIZE
                    )
                )

            elif self.axis == "y":
                rects.append(
                    pygame.Rect(
                        self.x * TILE_SIZE,
                        int((self.y + i * self.direction) * TILE_SIZE),
                        TILE_SIZE,
                        TILE_SIZE
                    )
                )

        return rects

    def get_players_on_top(self, players):
        tolerance = 5
        result = []

        for player in players:
            player_rect = player.rect()

            for rect in self.get_rects():

                # ONLY top contact matters
                horizontal_overlap = (
                    player_rect.right > rect.left and
                    player_rect.left < rect.right
                )

                on_top = abs(player_rect.bottom - rect.top) < tolerance

                if horizontal_overlap and on_top:
                    result.append(player)
                    break

        return result

    def can_extend(self, walls, door):
        if self.axis == "x":
            next_x = self.x + self.length * self.direction
            future_rect = pygame.Rect(
                int(next_x * TILE_SIZE),
                self.y * TILE_SIZE,
                TILE_SIZE,
                TILE_SIZE
            )
        
        else:  # axis == "y"
            next_y = self.y + self.length * self.direction
            future_rect = pygame.Rect(
                self.x * TILE_SIZE,
                int(next_y * TILE_SIZE),
                TILE_SIZE,
                TILE_SIZE
            )

        for wall in walls:
            if future_rect.colliderect(wall.rect()):
                return False

        if future_rect.colliderect(door.rect()):
            return False

        return True

    def update(self, group_action, players, walls, door, group):
        # --- STORE PREVIOUS POSITION (IMPORTANT FOR CARRYING) ---
        self.delta_x = 0
        self.delta_y = 0

        if not hasattr(self, "prev_x"):
            self.prev_x = self.x

        if not hasattr(self, "prev_y"):
            self.prev_y = self.y

        axis_action = group_action[self.axis]  # 1, -1, or 0

        players_on_top = self.get_players_on_top(players)

        if not players_on_top:
            self.prev_x = self.x
            self.prev_y = self.y
            return
        
        # --- GROUP INPUT DECISION (shared across extenders) ---
        group_extending = False
        group_retracting = False

        
        if axis_action == 1:
            if self.can_extend(walls, door):
                self.length += 1

        elif axis_action == -1:
            if self.length > self.min_length:

                # group constraint (all must agree)
                can_retract = True

                for other in group:
                    if other.length <= other.min_length:
                        can_retract = False
                        break

                if can_retract:
                    self.length -= 1
        # --- EXTEND ---
        if group_extending:
            if self.can_extend(walls, door):
                self.length += 1

                for player in players_on_top:
                    if self.axis == "x":
                        player.x += self.direction
                    elif self.axis == "y":
                        player.y += self.direction

        # --- RETRACT ---
        elif group_retracting:
            if self.length > self.min_length:

                can_retract = True

                for other in group:
                    if other.length <= other.min_length:
                        can_retract = False
                        break

                if can_retract:
                    self.length -= 1

        # --- CALCULATE MOVEMENT DELTA (THIS FRAME) ---
        self.delta_x = self.x - self.prev_x
        self.delta_y = self.y - self.prev_y

        # --- UPDATE STORED POSITION FOR NEXT FRAME ---
        self.prev_x = self.x
        self.prev_y = self.y

    def render(self, screen):
        for rect in self.get_rects():
            pygame.draw.rect(screen, self.color, rect)