import pygame
from env.entities import Player, Wall, Door, Extender
from env.entities.game_object import TILE_SIZE

WALL = 1
EMPTY = 0
DOOR = 3
EXTENDER_X = 4
EXTENDER_Y = 5
PLAYER = 9


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
        self.last_action = 2
        self.walls = []
        self.players = []
        self.extenders = []
        self.extender_groups = {
            "x": [],
            "y": []
        }

        self.door = None

        for y, row in enumerate(self.level):
            for x, tile in enumerate(row):
                if tile == WALL:
                    self.walls.append(Wall(x, y))
                elif tile == DOOR:
                    self.door = Door(x, y)
                elif tile == PLAYER:
                    self.players.append(Player(x, y))
                elif tile == EXTENDER_X:
                    e = Extender(x, y, "x", 1)
                    self.extenders.append(e)
                    self.extender_groups["x"].append(e)
                elif tile == EXTENDER_Y:  
                    e = Extender(x, y, "y", -1)
                    self.extenders.append(e)
                    self.extender_groups["y"].append(e)

        if len(self.players) == 0:
            raise ValueError("No player spawn (9) found in level")

        if self.door is None:
            raise ValueError("No door (3) found in level")

        self.done = False
        return self._get_state()

    def step(self, action, vertical_input=2):
        
        reward = -0.01

        self.last_action = action

        axis_pressure = {
            ("x", -1): 0,
            ("x", 1): 0,
            ("y", -1): 0,
            ("y", 1): 0
        }

        if self.last_action == 2:
            pass  # do nothing, but DO NOT exit function

        for player in self.players:
            player.update(action, self.walls, self.extenders)

        for player in self.players:
            if player.rect().colliderect(self.door.rect()):
                reward += 10
                self.done = True
                break


        group_action = {
            "x": 0,  # -1 retract, 1 extend
            "y": 0
        }

        for e in self.extenders:

            if action in (0, 1):
                # X axis controls
                if action == 1:
                    group_action["x"] = 1
                elif action == 0:
                    group_action["x"] = -1

            elif action in (3, 4):
                # Y axis controls
                if action == 3:
                    group_action["y"] = 1
                elif action == 4:
                    group_action["y"] = -1

        # update extenders
        for extender in self.extenders:
            extender.update(
                group_action,
                self.players,
                self.walls,
                self.door,
                self.extender_groups[extender.axis]
            )

        if self.render_mode:
            self._render()

        return self._get_state(), reward, self.done

    def _get_state(self):
        state = []
        for p in self.players:
            state.extend([p.x, p.vx])
        return state

    def _render(self):
        self.screen.fill((255, 255, 255))

        for extender in self.extenders:
            extender.render(self.screen)

        for wall in self.walls:
            wall.render(self.screen)

        self.door.render(self.screen)

        for player in self.players:
            player.render(self.screen)

        pygame.display.flip()
        self.clock.tick(10)

    def get_active_extenders(self):
        active = []
        for e in self.extenders:
            if len(e.get_players_on_top(self.players)) > 0:
                active.append(e)
        return active