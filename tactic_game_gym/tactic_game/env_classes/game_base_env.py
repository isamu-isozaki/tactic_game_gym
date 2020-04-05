from tactic_game_gym.tactic_game.env_classes.setup_env import Final_Var_Setup
from tactic_game_gym.tactic_game.utility import get_n_colors, get_network


import random, time, os, logging, pymunk, sys, time, cv2
from gym import spaces
import numpy as np
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import pygame
from pygame.locals import *

def sort_array(arr1, arr2):
    #sort the objects in arr1 according to the indices of arr2
    output = [None]*len(arr1)
    for i in arr2:
        output[i] = arr1[arr2[i]]
    return output


class Setup_Pygame_Pymunk(Final_Var_Setup):
    def __init__(self, **kwargs):
        Final_Var_Setup.__init__(self, **kwargs)
        if self.show:
            pygame.init()
            self.screen = pygame.display.set_mode(self.board_size)
            self.clock = pygame.time.Clock()
            self.surf = pygame.surfarray.make_surface(self.beautiful_map)
            if self.log:
                print(f"Finished generating pygame surface: {time.time()-self.start}")
    def switch_to_pymunk(self, xy):
        return [xy[0], -xy[1]+self.board_size[0]]
    def start_game(self):
        self.space = pymunk.Space()
        self.space.gravity = (0.0,0.0)
        self.balls = []


        self.lines = []
        self.masses = np.zeros(self.player_num, dtype=np.float16)
        mass = self.mass
        radius = mass**(1/3.)
        moment = pymunk.moment_for_circle(mass, 0, radius)
        colors = get_n_colors(n=self.sides*self.num_types)
        for i in range(self.sides):
            for j in range(self.players_per_side[i]):
                mass *= self.player_array[i][j].radius**3
                mass *= self.player_array[i][j].density
                if self.rand_params:
                    mass = np.random.normal(mass, self.rand_prop*mass)
                    if mass < 0:
                        mass = self.mass
                    mass *= self.player_array[i][j].radius**3#Initially, this player radius is just set to the proportion of the radius not the actual radius I guess

                radius = mass**(1/3.)
                moment = pymunk.moment_for_circle(mass, 0, radius)

                self.player_array[i][j].mass = mass
                self.player_array[i][j].radius = radius
                player = self.player_array[i][j]
                self.masses[player.id] = mass
                body = pymunk.Body(mass, moment)
                body.position = self.player_array[i][j].position[0], self.player_array[i][j].position[1]
                shape = pymunk.Circle(body, radius)
                # side_inverse = [2, 1]
                # mask = pymunk.ShapeFilter.ALL_MASKS ^ side_inverse[i]
                shape.filter = pymunk.ShapeFilter(group=2**player.side)
                shape.color = (*colors[i*self.num_types+player.type], 255)
                self.player_array[i][j].color = (*colors[i*self.num_types+player.type], 255)
                self.space.add(body, shape)
                self.balls.append(shape)
                mass = self.mass
        self.current_balls = self.balls.copy()
        #Add boundary cause I found that there may not be that much support for keeping stuff within a boundary
        body = pymunk.Body(body_type = pymunk.Body.STATIC)
        body.position = (0,0)
        wall_size = 4
        self.lines = [
        pymunk.Segment(body, (-wall_size,-wall_size), (-wall_size, self.board_size[1]+wall_size), wall_size),\
        pymunk.Segment(body, (-wall_size,-wall_size), (self.board_size[0]+wall_size, -wall_size), wall_size),\
        pymunk.Segment(body, (self.board_size[0]+wall_size,-wall_size), (self.board_size[0]+wall_size, self.board_size[1]+wall_size), wall_size),\
        pymunk.Segment(body, (-wall_size,self.board_size[1]+wall_size), (self.board_size[0]+wall_size, self.board_size[1]+wall_size), wall_size)
        ]
        self.space.add(*self.lines)
        
        self.started = True
    def interact_move(self, start_pos, end_pos, epsilon=1e-10):
        steps = self.vec_steps
        start_pos = np.array(self.switch_to_pymunk(start_pos), dtype=np.float16)
        end_pos = np.array(self.switch_to_pymunk(end_pos), dtype=np.float16)
        movement = end_pos-start_pos
        mag_div = self.board_size*self.vec_mag_div_constant_frac
        movement /= mag_div
        if np.sum(np.abs(movement)) > self.player_force_prop*2:
            mag = np.linalg.norm(movement)
            if mag > self.player_force_prop*np.sqrt(2):
                movement *= float(self.player_force_prop*np.sqrt(2))/mag
        size = self.vec_width#change with scrool or up and down arrow
        x = np.zeros(self.board_size[0], dtype=np.float16)
        y = np.zeros(self.board_size[0], dtype=np.float16)
        deduct_const = size // steps
        pos = start_pos
        pos =  np.asarray(pos, dtype=np.float16)
        while steps > 0:
            start = pos - size
            start[start < 0] = 0
            end = pos + size
            x[int(start[0]):int(end[0])]+=1
            y[int(start[1]):int(end[1])]+=1
            steps -= 1
            size -= deduct_const
        output = np.einsum("i,j->ij",x,y)
        output /= np.abs(output).max()
        valid_mask = output > 0
        try:
            self.move_board[self.interact_side,self.interact_type, valid_mask] += np.einsum("...,i->...i", output, movement)[valid_mask]
            self.move_board *= float(self.player_force_prop)/(np.abs(self.move_board+epsilon).max())
        except Exception as e:
            if self.log:
                print(f"{e}, move board shape: {self.move_board[self.interact_side,self.interact_type,valid_mask].shape}, output shape: {output.shape}, movement shape: {movement.shape}")
        if self.log:
            print(f"self.move_board mean: {self.move_board.mean()}, start position: {start_pos}, end position: {end_pos}")
    def clear_screen(self):
        """
        Call when clearing arrows. This occurs when space is pressed
        """
        self.move_board[self.interact_side,...] = 0
    def render_game(self):
        if self.show:
            self.screen.blit(self.surf, [0, 0])
            side = self.interact_side
            #self.screen.blit(self.surf, (0,0))
            #draw_width = self.draw_width
            for i in range(self.sides):
                for j in range(self.players_per_side[i]):
                    player = self.player_array[i][j]
                    if not player.alive or not (self.full_view or self.can_see[self.interact_side, player.id]):
                        continue
                    try:
                        x, y = player.vel
                        if np.isnan(y[0]):
                            y = x.copy()
                        x = self.switch_to_pymunk(x)
                        y = self.switch_to_pymunk(y)
                        hp = player.hp
                        opacity = hp/(self.hp*2)
                        opacity = 1 if opacity > 1 else opacity
                        opactiy = 0 if opacity is None else opacity
                        color_player = list(player.color[:3]) + [int(255*opacity)]
                        #This is a problem. Pygame only supports integers. Thus, animations won't be fluid
                        pygame.draw.circle(self.screen, color_player,self.switch_to_pymunk([int(player.position[0]), int(player.position[1])]), int(player.radius) if int(player.radius) > 0 else 1)
                        pygame.draw.line(self.screen, color_player, [int(x[0]), int(x[1])], [int(y[0]), int(y[1])])
                        if self.draw_connections:
                            for id in player.web:
                                if self.return_alive(id):
                                    y = self.board_sight[id, :2]
                                    y = self.switch_to_pymunk(y)
                                    pygame.draw.line(self.screen, color_player, [int(x[0]), int(x[1])], [int(y[0]), int(y[1])])
                    except Exception as e:
                        if self.log:
                            print(f"{e}. color: {player.color}. position: {player.position}, radius: {player.radius}, x: {x}, y: {y}")
                            import traceback

                            print(traceback.format_exc())


        self.space.step(self.game_timestep)
        if self.show:
            pygame.display.flip()

class Setup_Player_Graph(Setup_Pygame_Pymunk):
    def __init__(self, **kwargs):
        Setup_Pygame_Pymunk.__init__(self, **kwargs)
        self.det_heiarchy()
        if self.log:
            print(f"Finished setting heiarchy: {time.time()-self.start}")
    def det_heiarchy(self):
        #The heiarchy is determined by the sum of strength and hp
        aptitude = [self.stats[0][i] + self.stats[1][i] for i in range(self.sides)]

        aptitude_indices = [aptitude[i].argsort() for i in range(self.sides)]
        self.player_array =[sort_array(self.player_array[i], aptitude_indices[i]) for i in range(self.sides)]

        #Reset ids
        self.player_rank_id = np.zeros(self.player_num, dtype=np.float16)
        self.player_side = np.zeros(self.player_num, dtype=np.float16)
        k = 0
        for i in range(self.sides):
            for j in range(self.players_per_side[i]):
                self.player_array[i][j].id = k
                k+=1
        self.player_type_array = []
        self.player_type_ids = []#maybe useful later
        for i in range(self.sides):
            player_side_array = []
            id_side_array = []
            for j in range(self.num_types):
                type_array = []
                id_types = []
                for k in range(self.players_per_side[i]):
                    if self.player_array[i][k].type == j:
                        type_array.append(self.player_array[i][k])
                        id_types.append(int(self.player_array[i][k].id))
                id_types = np.asarray(id_types, dtype=np.float16)
                player_side_array.append(type_array)
                id_side_array.append(id_types)
            self.player_type_array.append(player_side_array)
            self.player_type_ids.append(id_side_array)
        networks = []
        for id_side_array in self.player_type_ids:
            side_network = []
            for id_type_array in id_side_array:
                side_network.append(get_network(id_type_array, self.num_subs))
            networks.append(side_network)
        for i in range(self.sides):
            for j in range(self.players_per_side[i]):
                player = self.player_array[i][j]
                graph = networks[i][player.type].pop(0)
                self.player_array[i][j].rank = graph["rank"]
                self.player_array[i][j].sub_ids = graph["subordinates"]
                self.player_array[i][j].superior_id = graph["superior"]
                sub_ids = self.player_array[i][j].sub_ids
                superior_id = self.player_array[i][j].superior_id
                self.player_array[i][j].web = (sub_ids if sub_ids != None else []) + ([superior_id] if superior_id != None else [])
                self.player_side[player.id] = i

class Set_Board(Setup_Player_Graph):
    def __init__(self, **kwargs):
        Setup_Player_Graph.__init__(self,**kwargs)
        self.set_board()
        if self.log:
            print(f"Finished setting up board: {time.time()-self.start}")
    def set_board(self):
        #The board is saved in a fashion where the number of channels is detemined by the number of sides
        self.board_sight = np.zeros([self.player_num, 18], dtype=np.float16)#holds all x position, y position, rank, side, alive
        for i in range(self.sides):
            for j in range(self.players_per_side[i]):
                position = self.player_array[i][j].position
                position = np.asarray(position, dtype=np.float16)
                position[position > self.board_size[0]-1] =  self.board_size[0]-1
                position = position.tolist()
                self.player_array[i][j].height = self.get_height(*position)
                self.player_array[i][j].cos = self.cos[int(position[0]), int(position[1])]
                self.player_array[i][j].sin = self.sin[int(position[0]), int(position[1])]
                player = self.player_array[i][j]
                k = player.id      
                self.board_sight[k, :2] = player.position
                self.board_sight[k, 2:4] = player.velocity
                self.board_sight[k, 4] = player.rank
                self.board_sight[k, 5] = player.side
                self.board_sight[k, 6] = player.strength
                self.board_sight[k, 7] = player.hp
                self.board_sight[k, 8] = player.mass
                self.board_sight[k, 9] = player.height
                try:
                    self.board_sight[k, 10] = player.cos
                    self.board_sight[k, 11] = player.sin
                except Exception as e:
                    if self.log:
                        print(f"Cosines: {player.cos}, Sine: {player.sin}")
                self.board_sight[k, 12] = player.base_vision
                try:
                    self.board_sight[k, 13:15] = self.move_board[int(player.side),int(player.type), int(position[0]),int(position[1])]
                except Exception as e:
                    if self.log:
                        print(f"{e}, move board shape: {self.move_board.shape}, side: {player.side}, position: {player.position}, index: {[player.side,player.position[0],player.position[1]]}")
                try:
                    self.board_sight[k, 15:17] = self.map_diff[:, int(position[0]), int(position[1])]
                except Exception as e:
                    print(f"{e}, map diff shape: {self.map_diff[:, int(position[0]), int(position[1])].shape}, side: {player.side}, position: {player.position}, index: {[player.side,player.position[0],player.position[1]]}, map diff value {self.map_diff[:, int(position[0]), int(position[1])]}")

                self.board_sight[k, 17] = player.alive
    def return_alive(self, ids):
        alive_index = 17
        return self.board_sight[ids, alive_index]
    def get_alive_mask(self):
        all_ids = np.arange(self.player_num)
        return self.return_alive(all_ids)

class Get_Sight(Set_Board):
    def __init__(self, **kwargs):
        Set_Board.__init__(self, **kwargs)
        self.get_sight()
    
    def get_sight(self, epsilon=1e-10):
        living = self.get_alive_mask() == 1

        vels = self.board_sight[living, 2:4].copy()
        strengths = self.board_sight[living, 6].copy()
        hps = self.board_sight[living, 7].copy()
        mass = self.board_sight[living, 8].copy()
        self.obs_full[..., 1:] = 0
        for i in range(self.sides):
            for i2 in range(self.sides):
                for j in range(self.players_per_side[i2]):
                    player = self.player_array[i2][j]
                    if not player.alive or not (self.full_view or self.can_see[i, player.id]):
                        continue
                    position = player.position
                    position = np.asarray(position, dtype=np.float16)
                    position[position > self.board_size[0]-1] =  self.board_size[0]-1
                    self.obs_full[i, int(position[0]), int(position[1]), 1] = player.hp if i == i2 else 0
                    self.obs_full[i, int(position[0]), int(position[1]), 2] = player.hp if i != i2 else 0
                    self.obs_full[i, int(position[0]), int(position[1]), 3:5] = player.velocity.copy() if i == i2 else [0,0]
                    self.obs_full[i, int(position[0]), int(position[1]), 5:7] = player.velocity.copy() if i != i2 else [0,0]
                    self.obs_full[i, int(position[0]), int(position[1]), 7] = self.attacked_dist[i, player.id]
            #normalize
            self.obs_full[i, np.abs(self.obs_full[i]) < epsilon] = 0
            self.obs_full[i, ..., 1:3] /= self.hp*(1+self.rand_prop)
            self.obs_full[i, ..., 3:7] /= self.max_speed
            self.obs_full[i, ..., 7] /= self.strength*self.max_players*(1-1/self.sides)*self.attack_div_frac
            self.obs_full [i, ..., 1:] *= 255
            #as in cnn, it's divided by 255
            """
            def print_obs_full_stats(m):
                data = self.obs_full[i, ..., m]
                size = self.obs_full[i, ..., m][self.obs_full[i, ..., m] != 0].shape
                print(f"obs {m} data: shape: {size[0]} max: {data.max()}, min: {data.min()}, mean: {data.mean()}, std: {data.std()}")
            for m in range(8):
                print_obs_full_stats(m)
            """
            resized_obs = cv2.resize(self.obs_full[i, ..., 1:].astype(np.float32), (self.obs_board_size, self.obs_board_size))
            self.obs[i, ..., 1:] = resized_obs.copy().astype(np.float16)
