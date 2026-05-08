import pygame
from .game_object import GameObject, TILE_SIZE


class Extender(GameObject):
    def __init__(self, x, y, axis, direction):
        if axis == "x":
            if direction == 1:
                super().__init__(x, y, (200, 100, 50))
            else:
                super().__init__(x, y, (200, 0, 50))
        else:
            super().__init__(x, y, (60, 232, 251))
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

    def collision_rects(self):
        return self.get_rects()

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

    def can_extend(self, walls, door, players, extenders, allowed_player_collisions=None):
        if allowed_player_collisions is None:
            allowed_player_collisions = []

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

        for extender in extenders:
            if extender is self:
                continue
        
            for rect in extender.collision_rects():
                
                if future_rect.colliderect(rect):
                    return False

        for player in players:
            if player in allowed_player_collisions:
                continue

            if future_rect.colliderect(player.rect()):
                return False

        return True

    def can_move_player(self, player, dx, dy, walls, door, players):
        future_rect = pygame.Rect(
            int((player.x + dx) * TILE_SIZE),
            int((player.y + dy) * TILE_SIZE),
            TILE_SIZE,
            TILE_SIZE
        )

        for wall in walls:
            if future_rect.colliderect(wall.rect()):
                return False

        if future_rect.colliderect(door.rect()):
            return False

        for other in players:
            if other is player:
                continue

            other_rect = other.rect()

            is_standing_on_other = (
                dy >= 0 and
                abs(future_rect.bottom - other_rect.top) <= 5 and
                future_rect.right > other_rect.left and
                future_rect.left < other_rect.right
            )

            if future_rect.colliderect(other_rect) and not is_standing_on_other:
                return False

        return True

    def action_for_group_action(self, group_action):
        axis_action = group_action[self.axis]

        if self.axis == "x":
            axis_action *= self.direction

        return axis_action

    def update(self, group_action, players, walls, door, group):
        axis_action = self.action_for_group_action(group_action)

        # IMPORTANT: check if ANY extender in group has a player on top
        players_on_top = self.get_players_on_top(players)

        # --- EXTEND ---
        if axis_action == 1:
            dx = 0
            dy = 0

            if self.axis == "y":
                dy = self.direction

            can_move_riders = all(
                self.can_move_player(player, dx, dy, walls, door, players)
                for player in players_on_top
            )

            if self.can_extend(
                walls,
                door,
                players,
                group,
                allowed_player_collisions=players_on_top
            ) and can_move_riders:
                self.length += 1

                for player in players_on_top:
                    if self.axis == "y":
                        player.y += dy

        # --- RETRACT ---
        elif axis_action == -1:
            if self.length > self.min_length:
                dx = 0
                dy = 0

                if self.axis == "y":
                    dy = -self.direction

                can_move_riders = all(
                    self.can_move_player(player, dx, dy, walls, door, players)
                    for player in players_on_top
                )

                if not can_move_riders:
                    return

                self.length -= 1

                for player in players_on_top:
                    if self.axis == "y":
                        player.y += dy

    def render(self, screen):
        for rect in self.get_rects():
            pygame.draw.rect(screen, self.color, rect)
