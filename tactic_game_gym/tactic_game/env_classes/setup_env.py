from tactic_game_gym.tactic_game.env_classes.map_env import Map_Env
from tactic_game_gym.tactic_game.player import Player
from tactic_game_gym.tactic_game.player_types.archer import Archer
from tactic_game_gym.tactic_game.player_types.cavarly import Cavarly
from tactic_game_gym.tactic_game.player_types.infantry import Infantry
from tactic_game_gym.tactic_game.player_types.wall import Wall

import random, time, os, time, cv2
from gym import spaces
import numpy as np

def sigmoid(x, derivative=False):
  return x*(1-x) if derivative else 1/(1+np.exp(-x))

# Classes to setup basic variables for future use

def get_random_normal_with_min(mean, std_prop, set_max=False, size=1):
    output = np.random.normal(mean, mean*std_prop,size)
    mask_min = output < mean*(1-std_prop)
    mask_max = False
    if set_max:
        mask_max = mean*(1+std_prop) < output
    output[mask_min] = mean*(1-std_prop)
    output[mask_max] = mean*(1+std_prop)
    return output

class Setup_Var_Init(Map_Env):
    def __init__(self, **kwargs):
        Map_Env.__init__(self, **kwargs)
        self.side = 0
        self.rotation = random.random() < 0.5
        self.t = 0
        self.attack_turn = 0
        self.base_directory = os.getcwd()

        self.started = False#check if game has started
        if self.is_train:
            import math
            if not hasattr(self, "total_moves"):
                self.total_moves = self.ended_moves
                self.stage = int(self.init_stage+self.ended_moves//self.stage_update_num)
        
        assert (self.rand_prop < 1 and 0 < self.rand_prop)
        self.player_force_prop /= np.sqrt(2)
        #For playing game
        self.vec_width = self.board_size[0]//self.sides
        self.interact_side = np.random.choice(np.arange(self.sides))
        if self.log:
            print(f"You are side {self.interact_side}")

        #numpy arrays below
        self.render_output = np.zeros([self.sides, self.obs_board_size, self.obs_board_size, 3], dtype=np.float16)
        self.finished_sides = np.zeros(self.sides, dtype=np.float16)
        self.move_board = np.zeros([self.sides] + self.board_size+[2], dtype=np.float16)
        obs_shape = (self.obs_board_size, self.obs_board_size, 1+2+2*2+1)
        obs_full_shape  = (*self.board_size, 1+2+2*2+1)
        self.obs = np.zeros([self.sides] + list(obs_shape), dtype=np.float16)
        self.obs_full = np.zeros([self.sides] + list(obs_full_shape), dtype=np.float16)
        for i in range(self.sides):
            self.obs_full[i, ...,  0] = self.map.copy()* 255
            self.obs[i, ...,  0] = cv2.resize(self.map.copy().astype(np.float32), (self.obs_board_size, self.obs_board_size)).astype(np.float16)* 255
        self.dead = np.zeros(self.sides, dtype=np.float16)#the amount that died during the current time step for all sides
        self.rewards = np.zeros(self.sides, dtype=np.float16)#the amount that died for other sides

        #Setting up observation and action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=[self.act_board_size*self.act_board_size*2], dtype=np.float32)
        self.action = np.zeros([self.sides, self.act_board_size, self.act_board_size, 2], dtype=np.float16)
        #1st screen: map(1), 2nd:hp(2) + 2*velocity(2), 3rd attack boards(1) 2 you or the enemy
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

class Set_Stats(Setup_Var_Init):
    def __init__(self, **kwargs):
        Setup_Var_Init.__init__(self, **kwargs)
        board_width = self.board_size[0]
        board_height = self.board_size[1]
        if self.rotation:
            board_width = self.board_size[1]
            board_height = self.board_size[0]
        width = (board_width)// self.sides
        players_per_side = [self.population_map[i*width:(i+1)*width].sum() for i in range(self.sides)]
        if self.rotation:
            players_per_side = [self.population_map[:, i*width:(i+1)*width].sum() for i in range(self.sides)]
        players_per_side = np.asarray(players_per_side, dtype=np.float16)
        players_per_side /= players_per_side.sum()
        players_per_side *= self.max_players
        players_per_side = players_per_side[::-1]
        if self.log:
            print(f"Got players per side: {time.time()-self.start}")
        self.k = [np.random.normal(loc=0, scale=self.std_k, size = int(players_per_side[i])) for i in range(self.sides)]
        self.k = [self.spring_force*sigmoid(self.k[i]) for i in range(self.sides)]
        if self.log:
            print(f"Finished generating k: {time.time()-self.start}")

        strength_stat = [get_random_normal_with_min(self.strength*self.moves_without_attack*self.game_timestep/0.2, self.cap_prop, size = int(players_per_side[i])) for i in range(self.sides)]
        energy_stat = [get_random_normal_with_min(self.hp, self.cap_prop, size = int(players_per_side[i])) for i in range(self.sides)]
        if self.log:
            print(f"Finished generating strength and energy: {time.time()-self.start}")


        self.stats = [strength_stat, energy_stat, self.k]
        self.k_ids = np.concatenate(self.k, axis=0)
        #sort into the three types: cavarly, archers and infantry
        if self.log:
            print(f"Finished generating stats: {time.time()-self.start}")

class Generate_Players(Set_Stats):
    def __init__(self, **kwargs):
        Set_Stats.__init__(self, **kwargs)
        self.player_array = []
        
        assert self.archer_prop + self.cavarly_prop + self.infantry_prop + self.wall_prop== 1
        probs = [self.archer_prop, self.cavarly_prop, self.infantry_prop, self.wall_prop]
        thresholds = [np.sum(probs[:i+1]) for i in range(len(probs))]
        classes = [Archer, Cavarly, Infantry, Wall]
        player_id = 0
        for i in range(self.sides):
            army = []
            for j in range(len(self.stats[0][i])):
                player = None
                random_val = random.random()
                for k, threshold in enumerate(thresholds):
                    if random_val < threshold:
                        army.append(classes[k](self.stats[0][i][j], self.stats[1][i][j], self.stats[2][i][j], player_id, i, **self.kwargs))
                        player_id += 1
                        break
            self.player_array.append(army)

        self.remaining_players = [len(self.player_array[i]) for i in range(self.sides)]
        self.players_per_side = np.copy(self.remaining_players)#contains the original players per side
        self.player_num = player_id#number of players+2
        self.player_forces = np.zeros([self.player_num], dtype=np.float16)
        self.r_as = np.ones([self.player_num])
        self.player_type_mask = np.zeros(self.player_num, dtype=np.uint8)
        player_types = ["archer", "cavarly", "infantry", "wall"]
        for i in range(self.sides):
            for j in range(self.players_per_side[i]):
                self.player_forces[self.player_array[i][j].id] = self.player_array[i][j].player_force
                self.r_as[self.player_array[i][j].id] = self.player_array[i][j].r_a
                self.player_type_mask[self.player_array[i][j].id] = player_types.index(self.player_array[i][j].player_name)
        
        if self.log:
            print(f"Finished generating players: {time.time()-self.start}")

class Post_Player_Setup(Generate_Players):
    def __init__(self, **kwargs):
        Generate_Players.__init__(self, **kwargs)
        self.attacked = np.zeros([self.sides, self.player_num], dtype=np.float16)#Damaged inflicted to side i
        self.attacked_dist = np.zeros([self.sides, self.player_num], dtype=np.float16)#Damage applied at location of players from side i include friendly fire
        self.can_see = np.zeros([self.sides, self.player_num], dtype=np.float16)
        self.web_mat = np.zeros([self.player_num, 2*self.num_subs-1, 2], dtype=np.float16)
        self.mag_mat = np.zeros([self.player_num, 2*self.num_subs-1], dtype=np.float16)
        self.k_mat = np.zeros([self.player_num, 2*self.num_subs-1], dtype=np.float16)
        self.m_mat = np.zeros([self.player_num, 2*self.num_subs-1], dtype=np.float16)
        

class Set_Player_Locs(Post_Player_Setup):
    def __init__(self, **kwargs):
        Post_Player_Setup.__init__(self, **kwargs)
        try:
            board_width = self.board_size[0]
            board_height = self.board_size[1]
            if self.rotation:
                board_width = self.board_size[1]
                board_height = self.board_size[0]
            width = (board_width)// self.sides
            for i in range(self.sides):
                p_density = self.population_map[i*width: (i+1)*width, :].copy()
                if self.rotation:
                    p_density = self.population_map[:, i*width: (i+1)*width].copy()
                p_density = np.reshape(p_density, [-1])
                p_density /= p_density.sum()
                locations = np.random.choice(np.arange(p_density.shape[0]), size=p_density.shape[0], replace=False, p=p_density)
                used = np.zeros(p_density.shape[0], dtype=np.uint8)
                #The number of players is self.players_per_side[i]
                k = 0
                for j in range(self.players_per_side[i]):#first index is player, next is location
                    while used[k] == 1 or (locations[k]+1) % board_height == 0 or locations[k] % board_height == 0 or locations[k] // board_height == 0:#Stops players from occupying corners
                        used[k] = 1
                        k+=1
                    used[k] = 1
                    location = np.asarray([width*i + locations[k] // board_height, locations[k] % board_height], dtype=np.float16)

                    if self.rotation:
                        location = np.asarray([locations[k] % board_height, width*i + locations[k] // board_height], dtype=np.float16)

                    used[np.where(locations == locations[k] + board_height)] = 1
                    used[np.where(locations == locations[k] + board_height+1)] = 1
                    used[np.where(locations == locations[k] + board_height-1)] = 1
                    used[np.where(locations == locations[k] - board_height)] = 1
                    used[np.where(locations == locations[k] - board_height+1)] = 1
                    used[np.where(locations == locations[k] - board_height-1)] = 1
                    used[np.where(locations == locations[k] +1)] = 1
                    used[np.where(locations == locations[k] -1)] = 1
                    self.player_array[i][j].position = location
                    #Thus each player will be set in a unique location
        except Exception as e:
            if self.log:
                print(f"{e}")
            self.__init__(**self.kwargs)
        if self.log:
            if self.log:
                print(f"Finished placing players: {time.time()-self.start}")

class Final_Var_Setup(Set_Player_Locs):
    def __init__(self, **kwargs):
        Set_Player_Locs.__init__(self, **kwargs)
        self.webs = []
        for _ in range(self.player_num):
            self.webs.append(None)
        self.mags = []
        for _ in range(self.player_num):
            self.mags.append(None)