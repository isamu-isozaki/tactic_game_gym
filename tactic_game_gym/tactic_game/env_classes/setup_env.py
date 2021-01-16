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
        self.init_reward = True
        self.side = 0
        self.rotation = random.random() < 0.5
        self.flip = random.random() < 0.5
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
            player_types = ["archer", "cavarly", "infantry", "wall"]
            print(f"You are side {self.interact_side}")

        #numpy arrays below
        self.render_output = np.zeros([self.sides, self.obs_board_size, self.obs_board_size, 3], dtype=np.float32)
        self.finished_sides = np.zeros(self.sides, dtype=np.float32)
        self.move_board = np.zeros([self.sides] + [self.num_types] + self.board_size+[2], dtype=np.float32)
        # obs_shape = (self.obs_board_size, self.obs_board_size, 1+2)
        # obs_full_shape  = (*self.board_size, 1+2)
        obs_shape = (self.obs_board_size, self.obs_board_size, 2*3)
        obs_full_shape  = (*self.board_size, 2*3)
        self.obs = np.zeros([self.sides] + list(obs_shape), dtype=np.float32)
        self.obs_full = np.zeros([self.sides] + list(obs_full_shape), dtype=np.float32)
        # for i in range(self.sides):
        #     self.obs_full[i, ...,  0] = (self.map.copy()-self.map.mean())
        #     self.obs_full[i, ..., 0] *= 255/self.obs_full[i, ..., 0].max()
        #     self.obs[i, ...,  0] = cv2.resize(self.obs_full[i, ..., 0].astype(np.float32), (self.obs_board_size, self.obs_board_size)).astype(np.float32)
        self.dead = np.zeros(self.sides, dtype=np.float32)#the amount that died during the current time step for all sides
        self.rewards = np.zeros(self.sides, dtype=np.float32)#the amount that died for other sides
        self.hard_coded_rewards = {'r_death_offset': np.zeros(self.sides, dtype=np.float32), 'r_damage': np.zeros(self.sides, dtype=np.float32), 'r_seen': np.zeros(self.sides, dtype=np.float32)}

        #Setting up observation and action space
        self.action_space = spaces.MultiDiscrete([5 for _ in range(self.act_board_size*self.act_board_size*self.num_types)])
        self.action = np.zeros([self.sides, self.act_board_size, self.act_board_size, self.num_types, 2], dtype=np.float32)
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
        players_per_side = [1/self.sides for i in range(self.sides)]
        players_per_side = np.asarray(players_per_side, dtype=np.float32)
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
        
        #Make it so that order to make prop 0 doesn't matter
        self.class_probs = np.array([self.archer_prop, self.cavarly_prop, self.infantry_prop, self.wall_prop])
        class_nums = [0, 1, 2, 3]
        valid_class_mask = self.class_probs != 0
        self.index_to_type = np.array(class_nums)[valid_class_mask]
        self.type_to_index = {self.index_to_type[k]: k for k in range(self.num_types)}
        self.interact_type = self.index_to_type[np.random.choice(np.arange(self.num_types))]
        if self.log:
            player_types = ["archer", "cavarly", "infantry", "wall"]
            print("You are ", player_types[self.interact_type])
        self.class_probs = self.class_probs[valid_class_mask]
        self.class_thresholds = [np.sum(self.class_probs[:i+1]) for i in range(self.num_types)]
        classes = np.array([Archer, Cavarly, Infantry, Wall])
        classes = classes[valid_class_mask]
        player_id = 0
        self.wall_nums = [0 for _ in range(self.sides)]
        for i in range(self.sides):
            army = []
            for j in range(len(self.stats[0][i])):
                player = None
                random_val = random.random()
                for k, threshold in enumerate(self.class_thresholds):
                    if random_val < threshold:
                        if k == 3:
                            self.wall_nums[i] += 1
                        army.append(classes[k](self.stats[0][i][j], self.stats[1][i][j], self.stats[2][i][j], player_id, i, **self.kwargs))
                        player_id += 1
                        break
            self.player_array.append(army)
        self.remaining_players = [len(self.player_array[i]) for i in range(self.sides)]
        self.players_per_side = np.copy(self.remaining_players)#contains the original players per side
        self.player_num = player_id#number of players+2
        self.player_forces = np.zeros([self.player_num], dtype=np.float32)
        self.player_sides = np.zeros([self.player_num], dtype=np.float32)
        self.r_as = np.ones([self.player_num])
        self.player_type_mask = np.zeros(self.player_num, dtype=np.uint8)
        for i in range(self.sides):
            for j in range(self.players_per_side[i]):
                self.player_forces[self.player_array[i][j].id] = self.player_array[i][j].player_force
                self.r_as[self.player_array[i][j].id] = self.player_array[i][j].r_a
                self.player_type_mask[self.player_array[i][j].id] = self.player_array[i][j].type
                self.player_sides[self.player_array[i][j].id] = i
        
        if self.log:
            print(f"Finished generating players: {time.time()-self.start}")

class Post_Player_Setup(Generate_Players):
    def __init__(self, **kwargs):
        Generate_Players.__init__(self, **kwargs)
        self.attacked = np.zeros([self.sides, self.player_num], dtype=np.float32)#Damaged inflicted to side i
        self.attacked_dist = np.zeros([self.sides, self.player_num], dtype=np.float32)#Damage applied at location of players from side i include friendly fire
        self.can_see = np.zeros([self.sides, self.player_num], dtype=np.float32)
        self.web_mat = np.zeros([self.player_num, 2*self.num_subs-1, 2], dtype=np.float32)
        self.mag_mat = np.zeros([self.player_num, 2*self.num_subs-1], dtype=np.float32)
        self.k_mat = np.zeros([self.player_num, 2*self.num_subs-1], dtype=np.float32)
        self.m_mat = np.zeros([self.player_num, 2*self.num_subs-1], dtype=np.float32)
        

class Set_Player_Locs(Post_Player_Setup):
    def __init__(self, **kwargs):
        Post_Player_Setup.__init__(self, **kwargs)
        player_type_order = np.array([3, 2, 0, 1])
        player_props = np.array([self.wall_prop, self.infantry_prop, self.archer_prop, self.cavarly_prop])
        player_type_order = player_type_order[player_props != 0]
        #"walls", "infantry", "archer", "cavarly"
        
        try:
            board_width = self.board_size[0]
            board_height = self.board_size[1]
            if self.rotation:
                board_width = self.board_size[1]
                board_height = self.board_size[0]
            width = (board_width)// (self.sides*self.num_types)
            for i in range(self.sides*self.num_types):
                p_density = self.population_map[i*width: (i+1)*width, :].copy()
                if self.rotation:
                    p_density = self.population_map[:, i*width: (i+1)*width].copy()
                p_density = np.reshape(p_density, [-1])
                p_density /= p_density.sum()
                locations = np.random.choice(np.arange(p_density.shape[0]), size=p_density.shape[0], replace=False, p=p_density)
                #The number of players is self.players_per_side[i]
                k = 0
                #assume only 2 players
                if i < self.num_types:
                    current_type = player_type_order[::-1][i]
                else:
                    current_type = player_type_order[i % self.num_types]

                for j in range(self.players_per_side[i//self.num_types]):#first index is player, next is location
                    if self.player_array[i//self.num_types][j].type != current_type:
                        continue
                    k += 1
                    location = np.asarray([width*i + locations[k] // board_height, locations[k] % board_height], dtype=np.float32)
                    if self.flip:
                        location[0] = board_width - location[0]

                    if self.rotation:
                        location = np.asarray([locations[k] % board_height, width*i + locations[k] // board_height], dtype=np.float32)
                        if self.flip:
                            location[1] = board_width - location[1]
                    self.player_array[i//self.num_types][j].set_position(location)
                    #Thus each player will be set in a unique location
        except Exception as e:
            if self.log:
                print(f"{e}")
                import traceback
                print(traceback.format_exc())
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
