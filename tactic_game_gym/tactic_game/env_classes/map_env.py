from tactic_game_gym.tactic_game.env_classes.args_env import Args_Env
from tactic_game_gym.tactic_game._map_generating_methods import diamond_square

import time, time, cv2
import numpy as np


class Map_Env(Args_Env):
    def __init__(self, **kwargs):
        Args_Env.__init__(self, **kwargs)
        self.start = time.time()
        self.board_size = [self.board_size, self.board_size]
        self.map = self.get_map()
        if self.log:
            print(f"Finished generating map: {time.time()-self.start}")
        self.beautiful_map = self.map.copy()
        self.beautiful_map = self.beautiful_map[:, ::-1]
        self.beautiful_map -= self.beautiful_map.min()
        self.beautiful_map *= 255/self.beautiful_map.max()
        self.beautiful_map = cv2.applyColorMap(self.beautiful_map.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        self.beautiful_map = cv2.cvtColor(self.beautiful_map, cv2.COLOR_RGB2BGR)
        self.beautiful_output = np.copy(self.beautiful_map)
        self.beautiful_output = np.stack([self.beautiful_output for _ in range(self.sides)])

        self.map_diff = [np.diff(self.map.copy(), axis=1, append=0.)[None],np.diff(self.map.copy(), axis=0, append=0.)[None]]
        self.map_diff = np.concatenate(self.map_diff, axis=0)
        self.angle_map = self.map_diff.copy()
        self.angle_map *= np.tan(self.max_angle)/self.angle_map.max()
        self.angle_map = np.arctan(self.angle_map)#self.angle_map[0, coords] gives angle across
        self.angle_map = np.mean(self.angle_map, axis=0)

        self.cos = np.cos(self.angle_map)
        self.sin = np.sin(self.angle_map)
        if self.log:
            print(f"Got angle map: {time.time() - self.start}")
        self.population_map = self.get_map()
        if self.log:
            print(f"Finished generating population map: {time.time()-self.start}")

    def get_map(self):
        coordinates = diamond_square.DiamondSquare(self.map_board_size)
        coordinates = np.asarray(coordinates, dtype=np.float16)
        coordinates = np.reshape(coordinates, [self.map_board_size, self.map_board_size]+ [3])
        coordinates = coordinates[:,:,-1]/256.0
        coordinates = cv2.resize(coordinates.astype(np.float32), tuple(self.board_size), interpolation=cv2.INTER_CUBIC).astype(np.float16)
        return coordinates
    def get_height(self, x, y):
        x1 = int(x)
        y1 = int(y)
        x1 = self.board_size[0]-1 if x1 >= self.board_size[0] else x1
        y1 = self.board_size[1]-1 if y1 >= self.board_size[1] else y1

        x2 = x1 + 1
        y2 = y1 + 1
        x2 = x1 if x2 >= self.board_size[0] else x2
        y2 = y1 if y2 >= self.board_size[1] else y2
        h0 = self.map[x1, y1]
        h1 = self.map[x2, y1]
        h2 = self.map[x1, y2]
        h3 = self.map[x2, y2]
        if x1 == x2 and y2 == y1:
            return h0
        x_ = x - x1
        y_ = y - y1
        h = ((h0*(1-x_)+h1*x_)*(1-y_)+(h2*(1-x_)+h3*x_)*y_)
        return h