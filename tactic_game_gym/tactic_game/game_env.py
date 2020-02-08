from tactic_game_gym.tactic_game.env_base import Base_Env
from tactic_game_gym.tactic_game.player import Player
from tactic_game_gym.tactic_game.game_args import game_args_parser
from tactic_game_gym.tactic_game._map_generating_methods import diamond_square

import random, time, os, logging, pymunk, sys, time, cv2
from gym import spaces
import numpy as np
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import pygame
from pygame.locals import *


def sigmoid(x, derivative=False):
  return x*(1-x) if derivative else 1/(1+np.exp(-x))
def describe(x):
	x = np.array(x)
	print(f"shape: {x.shape}, mean: {x.mean()}, std: {x.std()}, min: {x.min()}, max: {x.max()}")

class Game_Env_v0(Base_Env):
	def __init__(self, **kwargs):
		arg_parser = game_args_parser()
		args, _ = arg_parser.parse_known_args()
		self.kwargs = vars(args)
		self.kwargs.update(kwargs)
		for k,v in self.kwargs.items():
			setattr(self, k, v)
		
		#Thanks https://stackoverflow.com/questions/5624912/kwargs-parsing-best-practice
		self.side = 0
		self.start = time.time()
		if self.is_train:
			import math
			if not hasattr(self, "total_moves"):
				self.total_moves = self.ended_moves
				self.stage = int(self.init_stage+self.ended_moves//self.stage_update_num)
				self.num_stages = int(math.log(self.act_board_size, 2) - math.log(self.stage, 2))

		self.render_output = np.zeros([self.sides, self.obs_board_size, self.obs_board_size, 3])
		if self.show:
			pygame.init()
			self.screen = pygame.display.set_mode([self.board_size, self.board_size])
			self.clock = pygame.time.Clock()
		self.started = False#check if game has started
		self.finished_sides = np.zeros(self.sides)
		self.t = 0
		super(Game_Env_v0, self).__init__()
		self.attack_turn = 0
		assert (self.rand_prop < 1 and 0 < self.rand_prop)
		self.board_size = [self.board_size, self.board_size]
		self.player_force_prop /= np.sqrt(2)
		self.vec_width = self.board_size[0]//self.sides#width or area of effect of vector

		self.base_directory = os.getcwd()
		training = not self.test

		def get_random_normal_with_min(mean, std_prop, size=1):
			output = np.random.normal(mean, mean*std_prop,size)
			mask = output < mean*(1-std_prop)
			output[mask] =  mean*(1-std_prop)
			return output

		self.move_board = np.zeros([self.sides] + self.board_size+[2])

		board = np.zeros(self.board_size)
		rotation = random.random() < 0.5
		self.map = self.get_map()
		if self.log:
			print(f"Finished generating map: {time.time()-self.start}")
		self.population_map = self.get_map()
		if self.log:
			print(f"Finished generating population map: {time.time()-self.start}")

		#self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.act_board_size, self.act_board_size, 2), dtype=np.float32)
		#The action space that I wanted
		self.action_space = spaces.Box(low=-1.0, high=1.0, shape=[self.act_board_size*self.act_board_size*2], dtype=np.float32)
		self.action = np.zeros([self.sides, self.act_board_size, self.act_board_size, 2])
		#1st screen: map(1), 2nd:hp(2) + 2*velocity(2), 3rd attack boards(1) 2 you or the enemy
		obs_shape = (self.obs_board_size, self.obs_board_size, 1+2+2*2+1)
		obs_full_shape  = (*self.board_size, 1+2+2*2+1)
		self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
		self.obs = np.zeros([self.sides] + list(obs_shape))
		self.obs_full = np.zeros([self.sides] + list(obs_full_shape))
		#setting first index to map
		for i in range(self.sides):
			self.obs_full[i, ...,  0] = self.map.copy()* 255
			self.obs[i, ...,  0] = cv2.resize(self.map.copy(), (self.obs_board_size, self.obs_board_size))* 255
		
		self.beautiful_map = self.map.copy()
		self.beautiful_map = self.beautiful_map[:, ::-1]
		self.beautiful_map -= self.beautiful_map.min()
		self.beautiful_map *= 255/self.beautiful_map.max()
		self.beautiful_map = cv2.applyColorMap(self.beautiful_map.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
		self.beautiful_map = cv2.cvtColor(self.beautiful_map, cv2.COLOR_RGB2BGR)
		self.beautiful_output = np.copy(self.beautiful_map)
		self.beautiful_output = np.stack([self.beautiful_output for _ in range(self.sides)])
		if self.show:
			self.surf = pygame.surfarray.make_surface(self.beautiful_map)
			if self.log:
				print(f"Finished generating pygame surface: {time.time()-self.start}")
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
		board_width = self.board_size[0]
		board_height = self.board_size[1]
		if rotation:
			board_width = self.board_size[1]
			board_height = self.board_size[0]
		width = (board_width)// self.sides
		players_per_side = [self.population_map[i*width:(i+1)*width].sum() for i in range(self.sides)]
		if rotation:
			players_per_side = [self.population_map[:, i*width:(i+1)*width].sum() for i in range(self.sides)]
		players_per_side = np.asarray(players_per_side)
		players_per_side /= players_per_side.sum()
		players_per_side *= self.max_players
		players_per_side = players_per_side[::-1]
		if self.log:
			print(f"Got players per side: {time.time()-self.start}")


		self.k = [np.random.normal(loc=0, scale=self.std_k, size = int(players_per_side[i])) for i in range(self.sides)]
		self.k = [self.spring_force*sigmoid(self.k[i]) for i in range(self.sides)]
		#for i in range(self.sides):
		#      self.k[i][...] = 1.#set to 1 for now
		#Denotes trust
		if self.log:
			print(f"Finished generating k: {time.time()-self.start}")

		strength_stat = [get_random_normal_with_min(self.strength*self.moves_without_attack*self.game_timestep/0.2, self.cap_prop, size = int(players_per_side[i])) for i in range(self.sides)]
		energy_stat = [get_random_normal_with_min(self.hp, self.cap_prop, size = int(players_per_side[i])) for i in range(self.sides)]
		if self.log:
			print(f"Finished generating strength and energy: {time.time()-self.start}")


		self.stats = [strength_stat, energy_stat, self.k]
		#sort into the three types: cavarly, archers and infantry
		if self.log:
			print(f"Finished generating stats: {time.time()-self.start}")

		self.player_array = []
		id = 1
		for i in range(self.sides):
			self.player_array.append([Player(self.stats[0][i][j], self.stats[1][i][j], self.stats[2][i][j], id+j, i) for j in range(len(self.stats[0][i]))])
			id += len(self.stats[0][i])
		assert(self.num_types == 3)
		self.colors = [[0,0,255], [0,255,0], [255,0,0]]


		self.remaining_players = [len(self.player_array[i]) for i in range(self.sides)]
		self.players_per_side = np.copy(self.remaining_players)#contains the original players per side
		self.player_num = id-1#number of players+2
		self.attacked = np.zeros([self.sides, self.player_num])#Damaged inflicted to side i
		self.attacked_dist = np.zeros([self.sides, self.player_num])#Damage applied at location of players from side i include friendly fire

		self.can_see = np.zeros([self.sides, self.player_num])

		self.dead = np.zeros(self.sides)#the amount that died during the current time step for all sides
		self.rewards = np.zeros(self.sides)#the amount that died for other sides

		self.k_ids = np.concatenate(self.k, axis=0)
		self.interact_side = np.random.choice(np.arange(self.sides))

		self.player_forces = np.zeros(self.player_num)
		self.web_mat = np.zeros([self.player_num, 2*self.num_subs-1, 2])
		self.mag_mat = np.zeros([self.player_num, 2*self.num_subs-1])
		self.k_mat = np.zeros([self.player_num, 2*self.num_subs-1])
		self.m_mat = np.zeros([self.player_num, 2*self.num_subs-1])
		self.r_as = np.ones([self.player_num])
		if self.log:
			print(f"Interact side: {self.interact_side}")
		for i in range(self.sides):
			cavarly_prop = 1
			archer_prop = 1
			while cavarly_prop+archer_prop >= 1 or cavarly_prop < 0 or archer_prop < 0:
				cavarly_prop = np.random.normal(1,self.rand_troop_prop)*self.cavarly_prop
				archer_prop = np.random.normal(1, self.rand_troop_prop)*self.archer_prop
			infantry_prop = 1 - cavarly_prop - archer_prop
			player_types = np.asarray([infantry_prop, archer_prop, cavarly_prop])
			for j in range(self.players_per_side[i]):
				index = np.random.choice(np.arange(self.num_types), p=player_types)
				self.player_array[i][j].type = index#0 is infantry, 1 ia archer and 2 is cavarly
				if self.rand_params:
					self.player_array[i][j].base_vision = get_random_normal_with_min(self.base_vision, self.rand_prop)
					self.player_array[i][j].player_force = get_random_normal_with_min(self.player_force, self.rand_prop)
					self.player_array[i][j].max_speed = get_random_normal_with_min(self.max_speed, self.rand_prop)
				else:
					self.player_array[i][j].base_vision = self.base_vision
					self.player_array[i][j].player_force = self.player_force
					self.player_array[i][j].max_speed = self.max_speed
				self.player_array[i][j].radius = 1.
				self.player_array[i][j].r_a = self.r_a
				self.player_array[i][j].force_prop = 1.

				if index == 1:
					self.player_array[i][j].base_vision *= self.archer_constant
					self.player_array[i][j].hp /= np.sqrt(self.archer_constant)
				elif index == 2:
					self.player_array[i][j].hp *= self.cavarly_hp
					self.player_array[i][j].radius *= self.cavarly_scale
					self.player_array[i][j].r_a *= self.cavarly_scale
					self.player_array[i][j].player_force *= self.cavarly_force
					self.player_array[i][j].force_prop = self.cavarly_force

					self.player_array[i][j].max_speed *= self.cavarly_max_speed
					self.player_array[i][j].base_vision += (self.cavarly_scale-1)#To compensate for larger size
					self.player_array[i][j].k *= self.cavarly_k
				self.player_array[i][j].r_a /= np.sqrt(2)
				self.player_forces[self.player_array[i][j].id-1] = self.player_array[i][j].player_force
				self.r_as[self.player_array[i][j].id-1] = self.player_array[i][j].r_a
			

		if self.log:
			print(f"Finished generating players: {time.time()-self.start}")

		#Given the self.board_size, and the number of sides, get the position of the players
		#Before we do anything, initially, it will simply be a random position within a given region
		#The region is determined simply by dividing the width of the board by the number of armies
		#TODO: implement a model which determines the initial position of players given their statistics
		#Under 0.3 is water region. force similar to friction works to prevent motion
		#Here, drag force is applied which is proportional to v^2 and in the opposite direction of velocity
		#The constant is 2 while v will be divided by 6 and squared so 0<v^2<1

		#Every turn 0.001 energy is consumed->1000 turns is the main limit of troops
		#The change in altitude causes mg in the downward dirction + only proportion given by angle is given as movement forward




		self.boundary_size = 4
		try:
			for i in range(self.sides):
				p_density = self.population_map[i*width: (i+1)*width, :].copy()
				if rotation:
					p_density = self.population_map[:, i*width: (i+1)*width].copy()
				p_density = np.reshape(p_density, [-1])
				p_density /= p_density.sum()
				locations = np.random.choice(np.arange(p_density.shape[0]), size=p_density.shape[0], replace=False, p=p_density)
				used = np.zeros(p_density.shape[0])
				#The number of players is self.players_per_side[i]
				k = 0
				for j in range(self.players_per_side[i]):#first index is player, next is location
					while used[k] == 1 or (locations[k]+1) % board_height == 0 or locations[k] % board_height == 0 or locations[k] // board_height == 0:#Stops players from occupying corners
						used[k] = 1
						k+=1
					used[k] = 1
					location = np.asarray([width*i + locations[k] // board_height, locations[k] % board_height])

					if rotation:
						location = np.asarray([locations[k] % board_height, width*i + locations[k] // board_height])

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
		"""
		Spaceout players so there are no collisons.
		"""
		if self.log:
			if self.log:
				print(f"Finished placing players: {time.time()-self.start}")


		for i in range(self.sides):
			for j in range(self.players_per_side[i]):
				self.player_array[i][j].set_params()
		#False | True -
		self.rotation = rotation
		self.webs = []
		for _ in range(self.player_num):
			self.webs.append(None)
		self.mags = []
		for _ in range(self.player_num):
			self.mags.append(None)
		self.det_heiarchy()
		if self.log:
			print(f"Finished setting heiarchy: {time.time()-self.start}")
		self.set_subordinates()
		if self.log:
			print(f"Finished setting subordinates: {time.time()-self.start}")
		self.set_board()
		if self.log:
			print(f"Finished setting up board: {time.time()-self.start}")
		self.get_sight()
	def switch_to_pymunk(self, xy):
		return [xy[0], -xy[1]+self.board_size[0]]
	def get_player_by_id(self, id):
		for i in range(self.sides):
			for j in range(self.players_per_side[i]):
				if self.player_array[i][j].id == id:
					return self.player_array[i][j], i, j
		return None, None, None
	def get_map(self):
		coordinates = diamond_square.DiamondSquare(self.map_board_size)
		coordinates = np.asarray(coordinates)
		coordinates = np.reshape(coordinates, [self.map_board_size, self.map_board_size]+ [3])
		coordinates = coordinates[:,:,-1]/256.0
		coordinates = cv2.resize(coordinates, tuple(self.board_size), interpolation=cv2.INTER_CUBIC)
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
	def get_n_colors(self, n=None):
		#n is self.sides
		output = []
		max_color = 256**3
		if n == None:
			n = self.sides
		interval = max_color / n
		colors = [interval*i for i in range(1, n+1)]
		bases = [256**2, 256, 1]
		for i in range(len(colors)):
			color = colors[i]
			rgb_color = []
			for j in range(3):
				color_comp = min(color // bases[j],256)
				color_comp -= 1
				color_comp = int(color_comp)
				rgb_color.append(color_comp)
				color -= color_comp*bases[j]
			output.append(rgb_color)
		output = np.asarray(output)
		return output
	def start_game(self):
		self.space = pymunk.Space()
		self.space.gravity = (0.0,0.0)
		self.balls = []


		self.lines = []
		self.masses = np.zeros(self.player_num)
		mass = self.mass

		radius = mass**(1/3.)
		moment = pymunk.moment_for_circle(mass, 0, radius)
		colors = self.get_n_colors()
		for i in range(self.sides):
			for j in range(self.players_per_side[i]):
				mass *= self.player_array[i][j].radius**3
				if self.rand_params:
					mass = np.random.normal(mass, self.rand_prop*mass)
					if mass < 0:
						mass = self.mass
					mass *= self.player_array[i][j].radius**3#Initially, this player radius is just set to the proportion of the radius not the actual radius I guess

				radius = mass**(1/3.)
				moment = pymunk.moment_for_circle(mass, 0, radius)
				colors = self.get_n_colors()

				self.player_array[i][j].mass = mass
				self.player_array[i][j].radius = radius
				self.masses[self.player_array[i][j].id-1] = mass
				body = pymunk.Body(mass, moment)
				body.position = self.player_array[i][j].position[0], self.player_array[i][j].position[1]
				shape = pymunk.Circle(body, radius)
				shape.color = (*colors[i], 255)
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
	def sort_array(self, arr1, arr2):
		#sort the objects in arr1 according to the indices of arr2
		output = [None]*len(arr1)
		for i in arr2:
			output[i] = arr1[arr2[i]]
		return output
	def det_heiarchy(self):
		#The heiarchy is determined by the sum of intelligence and strength
		#The array of the combined strength and intelligence is given by

		aptitude = [self.stats[0][i] + self.stats[1][i] for i in range(self.sides)]
		aptitude_indices = [aptitude[i].argsort() for i in range(self.sides)]

		#Thanks https://stackoverflow.com/questions/9007877/sort-arrays-rows-by-another-array-in-python
		self.player_array =[self.sort_array(self.player_array[i], aptitude_indices[i]) for i in range(self.sides)]



		#Reset ids
		self.player_rank_id = np.zeros(self.player_num)
		self.player_side = np.zeros(self.player_num)
		k = 1
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
						id_types.append(self.player_array[i][k].id)
				id_types = np.asarray(id_types)
				player_side_array.append(type_array)
				id_side_array.append(id_types)
			self.player_type_array.append(player_side_array)
			self.player_type_ids.append(id_side_array)

		sum_of_player_array = 0
		for i in range(self.sides):
			sum_of_player_array += len(self.player_array[i])
		sum_of_player_type_array = 0
		for i in range(self.sides):
			for j in range(self.num_types):
				sum_of_player_type_array += len(self.player_type_array[i][j])
		assert(sum_of_player_array == sum_of_player_type_array)
		#The players are sorted according to their aptitude

		self.player_ranks = [[np.ones(len(self.player_type_array[i][j])) for j in range(self.num_types)] for i in range(self.sides)]
		#The rank is determined by continuously having players ranked in the top 20% having a 1 higher rank than the rest

		#The minimum rank is 1
		rank_sizes = [[len(self.player_ranks[i][j])// self.num_subs for j in range(self.num_types)] for i in range(self.sides)]

		for i in range(self.sides):
			for j in range(self.num_types):
				while rank_sizes[i][j] > 0:
					self.player_ranks[i][j][:rank_sizes[i][j]] += 1
					rank_sizes[i][j] = rank_sizes[i][j] // self.num_subs
		#set player rank from what was determined before7->later
		m=0
		for i in range(self.sides):
			l = 0
			for j in range(self.num_types):
				for k in range(len(self.player_type_array[i][j])):
					self.player_array[i][l].rank =  self.player_ranks[i][j][k]
					self.player_type_array[i][j][k].rank = self.player_ranks[i][j][k]
					self.player_rank_id[m] = self.player_ranks[i][j][k]
					self.player_side[m] = self.player_array[i][l].side
					l += 1
					m += 1
	def set_subordinates(self):

		for i in range(self.sides):
			for m in range(self.num_types):
				try:
					ranks = np.arange(start= self.player_ranks[i][m][0], stop = 1, step = -1)#All the ranks in the reverse order
				except Exception as e:
					if self.log:
						print(f"{e}. ranks shape: {ranks.shape}. ranks: {ranks}")
				#Does not include the rank of 1 as the lowest level does not have subordinates
				for rank in ranks:
					same_rank = np.where(self.player_ranks[i][m] == rank)[0]
					#the indices of players with the given rank in player_rank[i]
					lower_rank = np.where(self.player_ranks[i][m] == rank-1)[0]

					same_rank_size = len(same_rank)
					lower_rank_size = len(lower_rank)

					#The highest aptitude in the same rank has the most number of subordinates if there are remainders
					sub_num = lower_rank_size//same_rank_size
					#The number of subordinates when not considering the remainder
					sub_rem = lower_rank_size % same_rank_size
					#the remainder

					#The higher the aptitude, the higher the quality of troops as well
					#the highest gets the highest number of troops as well
					current_j = 0
					for k in range(0, same_rank_size):#0 already done above
						if sub_rem > 0:
							self.player_type_array[i][m][same_rank[k]].sub_ids = [self.player_type_array[i][m][lower_rank[j]].id for j in range(current_j, current_j + sub_num+1)]
							self.player_type_array[i][m][same_rank[k]].sub_js = [lower_rank[j] for j in range(current_j, current_j + sub_num+1)]
							current_j += (sub_num+1)
							sub_rem -= 1
						else:
							self.player_type_array[i][m][same_rank[k]].sub_ids = [self.player_type_array[i][m][lower_rank[j]].id for j in range(current_j, current_j + sub_num)]
							self.player_type_array[i][m][same_rank[k]].sub_js = [lower_rank[j] for j in range(current_j, current_j + sub_num)]
							current_j += sub_num
					for k in range(same_rank_size):
						for j in self.player_type_array[i][m][same_rank[k]].sub_js:
							self.player_type_array[i][m][j].superior_id = self.player_type_array[i][m][same_rank[k]].id
							self.player_type_array[i][m][j].superior_j = same_rank[k]
			unused_k = np.arange(self.players_per_side[i])
			used_indices = []

			for m in range(self.num_types):
				for k in range(len(unused_k)):
					player = self.player_array[i][unused_k[k]]
					try:
						index = np.where(player.id == self.player_type_ids[i][m])[0]
						if len(index) == 0:
							k+=1
							continue
						used_indices.append(k)
						index = index[0]
						self.player_array[i][unused_k[k]].sub_ids = self.player_type_array[i][m][index].sub_ids
						self.player_array[i][unused_k[k]].sub_js = self.player_type_array[i][m][index].sub_js
						self.player_array[i][unused_k[k]].superior_id = self.player_type_array[i][m][index].superior_id
						self.player_array[i][unused_k[k]].superior_j = self.player_type_array[i][m][index].superior_j
					except Exception as e:
						if self.log:
							print(f"{e}, i: {i}, k: {k}, m: {m}, index: {index}, id: {player.id}")
				used_indices = np.asarray(used_indices)
				while len(used_indices) > 0:
					index = used_indices[0]
					used_indices = used_indices[1:]
					unused_k = np.concatenate([unused_k[:index], unused_k[index+1:]], axis=0)
					if len(used_indices) > 0:
						used_indices -= 1
				used_indices = []
	def set_board(self):
		#The board is saved in a fashion where the number of channels is detemined by the number of sides
		self.board_sight = np.zeros([self.player_num, 18])#holds all x position, y position, rank, side, alive
		for i in range(self.sides):
			for j in range(self.players_per_side[i]):
				position = self.player_array[i][j].position
				position = np.asarray(position)
				position[position > self.board_size[0]-1] =  self.board_size[0]-1
				position = position.tolist()
				self.player_array[i][j].height = self.get_height(*position)
				self.player_array[i][j].cos = self.cos[int(position[0]), int(position[1])]
				self.player_array[i][j].sin = self.sin[int(position[0]), int(position[1])]

				player = self.player_array[i][j]
				k = player.id-1
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
					self.board_sight[k, 13:15] = self.move_board[int(player.side),int(position[0]),int(position[1])]
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
		return self.board_sight[ids-1, alive_index]
	def get_alive_mask(self):
		all_ids = np.arange(self.player_num) + 1
		return self.return_alive(all_ids)
	def get_web_and_mag(self, player):
		web = None
		if self.webs[player.id-1] is not None:
			web = self.webs[player.id-1]
		else:
			player_pos = self.board_sight[player.id-1, :2].copy()
			superior_pos = np.reshape(self.board_sight[player.superior_id-1, :2].copy(), (-1, 2)) if player.superior_id != None else None
			sub_pos = np.reshape(self.board_sight[np.asarray(player.sub_ids)-1, :2].copy(), (-1, 2)) if player.sub_ids != None else None

			if player.superior_id == None and player.sub_ids == None:
				return None, None

			if player.superior_id == None:
				web = sub_pos
				alive = self.return_alive(np.asarray(player.sub_ids))
				web = web[alive==1]
			elif player.sub_ids == None:
				web = superior_pos
				alive = self.return_alive(player.superior_id)
				if not alive:
					web = None
			else:
				try:
					web = np.concatenate([superior_pos, sub_pos], axis=0)
					alive = np.concatenate([np.asarray([self.return_alive(player.superior_id)]), self.return_alive(np.asarray(player.sub_ids))], axis=0)
					web = web[alive==1]
				except Exception as e:
					if self.log:
						print(f"{e}, superior shape: {superior_pos.shape}, sub_pos shape: {sub_pos.shape}, superior id alive: {self.return_alive(player.superior_id)}, sub id alive: {self.return_alive(np.asarray(player.sub_ids))}")

			if web is None or web.shape[0] == 0:
				web = None

			if web is not None:
				web -= player_pos
			self.webs[player.id -1] = web
		mag = None
		if web is None:
			return web, mag
		if self.mags[player.id-1] is not None:
			mag = self.mags[player.id-1]
		else:
			mag = np.linalg.norm(web, axis=1)
			self.mags[player.id -1] = mag
		return web, mag	
	def get_drag(self, player, force):
		if self.get_height(*player.position) != 0:
			return np.asarray([0,0])
		above_limit = np.abs(force[np.abs(force) > self.drag_force_prop])
		if above_limit.shape[0] is not 0:
			force /= above_limit.max()
			force *= self.drag_force_prop
		return force
	def rotate_force(self, player, force, z):
		"""
		Goal of this function:
		rotate the force and deduct/add force taken from gravity so that it makes it harder to climb up mountains and can speed up when going down mountains
		Current force-> 2d
		Convert force to -> 3d
		x -> x cos(angle)

		y -> y cos(angle)

		z -> root(x^2+y^2) sin(angle)

		Get unit vector by dividing by sqrt(x^2+y^2)

		cos alpha with the gravity will be the negative z component of the 3d vector

		multiply cos alpha by the mg scalar and multiply by the unit vector
		"""
		position = player.position
		position = np.asarray(position)
		position[position > self.board_size[0]-1] =  self.board_size[0]-1
		position = position.tolist()
		force_mag = np.linalg.norm(force)
		force_angles = None
		if force_mag == 0:
			force_angles = np.ones([2])/np.sqrt(2)
		else:
			force_angles = force/force_mag
		try:
			
			force_3d_unit = np.asarray([force_angles[0]*player.cos,\
			 force_angles[1]*player.cos,\
			z])
		except Exception as e:
			if self.log:
				print(f"{e}. force angles: {force_angles}, force: {force}")
		cos_weight = -force_3d_unit[2]
		cos_weight *= player.mass*self.g
		force_3d = force_3d_unit*(force_mag+cos_weight)
		return force_3d[:2]
	def interact_move(self, start_pos, end_pos, epsilon=1e-10):
		steps = self.vec_steps
		start_pos = self.switch_to_pymunk(start_pos)
		end_pos = self.switch_to_pymunk(end_pos)
		movement =np.asarray(end_pos) -  np.asarray(start_pos)
		movement = movement.astype(float)
		mag_div = self.board_size*self.vec_mag_div_constant_frac
		movement /= mag_div
		mag = np.linalg.norm(movement)
		if mag > self.player_force_prop*np.sqrt(2):
			movement *= float(self.player_force_prop*np.sqrt(2))/mag
		size = self.vec_width#change with scrool or up and down arrow
		x = np.zeros(self.board_size[0])
		y = np.zeros(self.board_size[0])
		deduct_const = size // steps
		pos = start_pos
		pos =  np.asarray(pos)
		while steps > 0:
			start = pos - size
			start[start < 0] = 0
			end = pos + size
			x[start[0]:end[0]]+=1
			y[start[1]:end[1]]+=1
			steps -= 1
			size -= deduct_const
		output = np.einsum("i,j->ij",x,y)
		output /= np.abs(output).max()
		valid_mask = output > 0
		try:
			self.move_board[self.interact_side,valid_mask] += np.einsum("...,i->...i", output, movement)[valid_mask]
			self.move_board *= float(self.player_force_prop)/(np.abs(self.move_board+epsilon).max())
		except Exception as e:
			if self.log:
				print(f"{e}, move board shape: {self.move_board[self.interact_side,valid_mask].shape}, output shape: {output.shape}, movement shape: {movement.shape}")
		if self.log:
			print(f"self.move_board mean: {self.move_board.mean()}, start position: {start_pos}, end position: {end_pos}")
	def clear_screen(self):
		"""
		Call when clearing arrows. This occurs when space is pressed
		"""
		self.move_board[self.interact_side,...] = 0
	def get_ks_ms(self, player):
		web, mag = self.get_web_and_mag(player)
		if web is None:
			return None, None
		player_velocity = self.board_sight[player.id-1, 2:4].copy()
		player_k = self.k_ids[player.id-1].copy()
		sub_k = self.k_ids[np.asarray(player.sub_ids)-1].copy() if player.sub_ids != None else None
		superior_mass = self.masses[player.superior_id-1].copy() if player.superior_id != None else None
		sub_masses = self.masses[np.asarray(player.sub_ids)-1].copy() if player.sub_ids != None else None
		ks = None
		ms = None
		if player.superior_id == None:
			ks = sub_k
			ms = sub_masses
			alive = self.return_alive(np.asarray(player.sub_ids))
			ks = ks[alive==1]
			ms = ms[alive==1]
		elif player.sub_ids == None:
			ks = np.array([player_k])
			ms = np.array([superior_mass])
		else:
			ks = np.concatenate([np.asarray([player_k]), sub_k], axis=0)
			ms = np.concatenate([np.asarray([superior_mass]), sub_masses], axis=0)
			alive = np.concatenate([np.asarray([self.return_alive(player.superior_id)]), self.return_alive(np.asarray(player.sub_ids))], axis=0)
			ks = ks[alive==1]
			ms = ms[alive==1]
		return ks, ms
	def get_spring_matrix(self):
		living = self.get_alive_mask() == 1
		self.k_mat[:] = 0
		self.m_mat[:] = 0
		self.web_mat[:] = 0
		self.mag_mat[:] = 0
		k = -1
		for i in range(self.sides):
			for j in range(self.players_per_side[i]):
				if not self.player_array[i][j].alive:
					continue
				k += 1
				player = self.player_array[i][j]
				id = player.id
				ks, ms = self.get_ks_ms(player)
				if ks is None or ms is None:	
					continue
				self.k_mat[k, :ks.shape[0]] = ks
				self.m_mat[k, :ms.shape[0]] = ms
				
				if self.webs[id-1] is None or self.mags[id-1] is None:
					continue
				self.web_mat[k, :self.webs[id-1].shape[0]] = self.webs[id-1]
				self.mag_mat[k, :self.mags[id-1].shape[0]] = self.mags[id-1]
	def get_springs(self):
		self.get_spring_matrix()
		living = self.get_alive_mask() == 1
		player_mass_mat = self.masses[living]
		mag_mat = self.mag_mat[living].copy()
		radiis = self.r_as[living].copy()
		mag_mat_mask = mag_mat == 0
		mag_mat[mag_mat_mask] = 1 
		radii_denom = 1/mag_mat
		radii_denom[mag_mat_mask] = 0
		radiis = np.einsum("i,i...->i...", radiis, radii_denom)#radiis has shape [num_living, 2*num_subs-1]
		radiis = 2*np.einsum("ij, ij...->ij...", radiis, self.web_mat[living])#web_mat has shape [num_living, 2*num_subs-1, 2]
		#k_mat has shape [num_living, 2*num_subs-1] as well as m_mat

		forces = np.einsum("ij,ij...->ij...", self.k_mat[living], self.web_mat[living]-radiis)
		mass_denom = np.einsum("i...,i->i...", self.m_mat[living], self.masses[living])

		#if mass denom evaluates to 0, as the mass of the superior/subordinates are set to be None, the k_mat which this is later
		#multiplied by will evaluate to 0 too. Thus, set to 1 to avoid complications
		mass_num = np.transpose(np.transpose(self.m_mat[living])+self.masses[living])
		mass_denom_mask = mass_denom == 0
		mass_denom[mass_denom_mask] = 1
		mass_denom = 1/mass_denom
		mass_denom[mass_denom_mask] = 0
		mass_factor = np.einsum("ij,ij->ij", mass_num, mass_denom)
		damping = -2*np.sqrt(np.einsum("ij,ij->ij", self.k_mat[living], mass_factor))
		damping = np.einsum("ij,ik->ijk", damping, self.board_sight[living, 2:4])
		forces += damping
		forces = np.sum(forces, axis=1)
		k = 0
		for i in range(self.sides):
			for j in range(self.players_per_side[i]):
				player = self.player_array[i][j]
				if not player.alive:
					continue
				force = forces[k]
				above_limit = np.abs(force[np.abs(force) > self.spring_force_prop*player.force_prop])
				if above_limit.shape[0] is not 0:
					force /= above_limit.max()
					force *= self.spring_force_prop
				forces[k] = force
				k += 1
		return forces
	def get_spring(self, player, epsilon=10**-50):
		web, mag = self.get_web_and_mag(player)
		if web is None:
			return np.asarray([0,0])
		player_velocity = self.board_sight[player.id-1, 2:4].copy()
		player_k = self.k_ids[player.id-1].copy()
		sub_k = self.k_ids[np.asarray(player.sub_ids)-1].copy() if player.sub_ids != None else None
		superior_mass = self.masses[player.superior_id-1].copy() if player.superior_id != None else None
		sub_masses = self.masses[np.asarray(player.sub_ids)-1].copy() if player.sub_ids != None else None
		ks = None
		ms = None
		if player.superior_id == None:
			ks = sub_k
			ms = sub_masses
			alive = self.return_alive(np.asarray(player.sub_ids))
			ks = ks[alive==1]
			ms = ms[alive==1]
		elif player.sub_ids == None:
			ks = player_k
			ms = superior_mass
		else:
			ks = np.concatenate([np.asarray([player_k]), sub_k], axis=0)
			ms = np.concatenate([np.asarray([superior_mass]), sub_masses], axis=0)
			alive = np.concatenate([np.asarray([self.return_alive(player.superior_id)]), self.return_alive(np.asarray(player.sub_ids))], axis=0)
			ks = ks[alive==1]
			ms = ms[alive==1]

		player_mass = self.masses[player.id-1].copy()
		mag[mag == 0] = 1
		try:
			radiis = np.diag(player.r_a/(mag)) @ web
		except Exception as e:
			if self.log:
				print(f"{e}, mag: {mag}, web: {web}")
		ks = np.reshape(ks, [-1])
		ms = np.reshape(ms, [-1])
		try:
			force = np.diag(ks) @(web-radiis*2)- 2*np.reshape(np.sqrt(ks*(ms+player_mass)/(ms*player_mass)), (-1, 1))@np.reshape(player_velocity, (1,2))
		except Exception as e:
			if self.log:
				print(f"{e}, web: {web}, web shape: {web.shape}, radiis: {radiis}, radiis shape: {radiis.shape}, ms: {ms}, ms shape: {ms.shape}, ks: {ks}, ks shape: {ks.shape}")
		force = np.sum(force, axis=0)
		above_limit = np.abs(force[np.abs(force) > self.spring_force_prop*player.force_prop])
		if above_limit.shape[0] is not 0:
			force /= above_limit.max()
			force *= self.spring_force_prop
		return force
	def move(self):
		living = self.get_alive_mask() == 1
		river = self.board_sight[:, 9] == 0
		is_drag = living & river
		forces = self.board_sight[living, 13:15].copy()
		forces = np.einsum("i..., i->i...", forces, self.player_forces[living])
		drag_forces = -np.einsum("i,i...->i...", self.vel_mags[is_drag], self.board_sight[is_drag, 2:4])
		spring_force = self.get_springs()
		forces += spring_force
		k = 0
		for i in range(self.sides):
			for j in range(self.players_per_side[i]):
				if not self.player_array[i][j].alive:
					continue
				player = self.player_array[i][j]
				force = forces[k]
				if is_drag[is_drag].shape[0] != 0:
					force += self.get_drag(player, drag_forces[drag_k])
				#force += self.get_spring(player)
				forces[k] = force
				if is_drag[living][k]:
					drag_k+=1
				k+=1

		z = np.abs(self.board_sight[living, 11].copy())#z index of rotated 3d array
		sign = np.sign(np.einsum("...i, ...i->...", forces, self.board_sight[living, 15:17]))
		z = np.einsum("i,i->i", z, sign)
		k = 0
		for i in range(self.sides):
			for j in range(self.players_per_side[i]):
				if not self.player_array[i][j].alive:
					continue
				player = self.player_array[i][j]
				force = forces[k]
				force = self.rotate_force(player, force, z[k])

				try:
					self.player_array[i][j].apply_force(force, self.current_balls[k])
				except Exception as e:
					if self.log:
						print(f"Exception: {e}, i: {i}, j: {j}, k: {k}, id: {player.id}, current_balls length: {len(self.current_balls)}")
				k += 1
		k = 0
		for i in range(self.sides):
			for j in range(self.players_per_side[i]):
				id = self.player_array[i][j].id
				if not self.player_array[i][j].alive:
					self.vel_mags[id-1] = 0
					continue
				position = self.current_balls[k].body._get_position()
				velocity = self.current_balls[k].body._get_velocity()
				speed = np.linalg.norm(velocity)
				scale = 1
				if speed > self.max_speed:
					scale = self.max_speed/speed
					self.current_balls[k].body._set_velocity((scale*np.asarray(velocity)).tolist())
					speed = self.max_speed
				self.player_array[i][j].speed = speed
				self.vel_mags[id-1] = speed
				self.player_array[i][j].vel = [self.player_array[i][j].position, self.player_array[i][j].position+scale*np.asarray(velocity)]
				self.player_array[i][j].position = position
				self.player_array[i][j].velocity = scale*np.asarray(velocity)
				k += 1


		self.set_board()#update the board
	def mobilize_step(self):
		self.move()
		self.render_game()
	def reset_web(self):
		self.webs = []
		for _ in range(self.player_num):
			self.webs.append(None)
		self.mags = []
		for _ in range(self.player_num):
			self.mags.append(None)
		self.vel_mags[...] = 0
	def mobilize(self):
		self.start_game()
		if self.log:
			print(f"Game setup completed at {time.time()-self.start}")
		if self.save_imgs:
			if not os.path.exists( self.base_directory   + "/animation"):
					os.makedirs( self.base_directory   + "/animation")
			folders = os.listdir(self.base_directory + "/animation")
		self.vel_mags = np.zeros(self.player_num)
		for t in tqdm(itertools.count()):
			try:
				for event in pygame.event.get():
					if event.type == MOUSEBUTTONUP:
						None
					elif event.type == pygame.QUIT:
						break
				self.mobilize_step()
				if self.save_imgs:
					self.show_board(folder=self.base_directory   + f"/animation/animation_{len(folders)}",save=self.save_animation, step=t, title="Moves of {}".format(self.remaining_players))
				self.reset_web()
				if t>=self.terminate_turn:
					break
			except KeyboardInterrupt:
				break
	def render_game(self):
		if self.show:
			self.screen.blit(self.surf, [0, 0])
			side = self.interact_side
			#self.screen.blit(self.surf, (0,0))
			#draw_width = self.draw_width
			colors = self.get_n_colors()
			for i in range(self.sides):
				color = colors[i]
				for j in range(self.players_per_side[i]):
					player = self.player_array[i][j]
					if not player.alive or not (self.full_view or self.can_see[self.interact_side, player.id-1]):
						continue
					try:
						x, y = player.vel
						x = self.switch_to_pymunk(x)
						y = self.switch_to_pymunk(y)
						hp = player.hp
						opacity = hp/(self.hp*2)
						opacity = 1 if opacity > 1 else opacity
						color_player = color.tolist() + [int(255*opacity)]
						#This is a problem. Pygame only supports integers. Thus, animations won't be fluid
						pygame.draw.circle(self.screen, color_player,self.switch_to_pymunk([int(player.position[0]), int(player.position[1])]), int(player.radius) if int(player.radius) > 0 else 1)
						pygame.draw.line(self.screen, color_player, [int(x[0]), int(x[1])], [int(y[0]), int(y[1])])
					except Exception as e:
						if self.log:
							print(f"{e}. color: {color}. position: {player.position}, radius: {player.radius}")


		self.space.step(self.game_timestep)
		if self.show:
			pygame.display.flip()
	def attack(self, epsilon=1e-5):
		rotation = True
		prop_side = self.prop_side
		positions = self.board_sight[:, :2].copy()
		velocities = self.board_sight[:, 2:4].copy()
		living = self.get_alive_mask() == 1
		alive = []
		for i in range(self.sides):
			for j in range(self.players_per_side[i]):
				if self.player_array[i][j].alive:
					alive.append(self.player_array[i][j].id)
		mags = self.vel_mags.copy()
		mags = mags[living]
		positions = positions[living]
		velocities = velocities[living]
		num_players = np.sum(self.remaining_players)
		attacked = np.zeros(num_players)
		cos_sin = (velocities+np.array([epsilon, epsilon])) / (mags[:, None]+epsilon*np.sqrt(2))
		rot_matrix = np.concatenate([\
							np.concatenate([cos_sin[:, 0, None, None],\
											 cos_sin[:, 1, None, None]], axis = 2),\
							np.concatenate([cos_sin[:, 1, None, None],\
											-cos_sin[:, 0, None, None]], axis = 2)], axis = 1)
		x, y = self.board_sight[living, 0], self.board_sight[living, 1]
		X = np.reshape(-positions[:, 0, None]+x, [num_players, num_players])#The x coordinate of all other players from each player
		Y = np.reshape(-positions[:, 1, None]+y, [num_players, num_players])
		#X[i], Y[i] is the x and y coordinates of all of the players when the ith player is at (0,0)
		XY = np.concatenate([X[..., None], Y[..., None]], axis = 2)
		attack_range_mags = mags / (self.attrange_div_const*self.max_speed)
		base_index = 12
		attack_range = self.range_factor*np.einsum("i,i->i",attack_range_mags, self.board_sight[living, base_index])[:, None]#set max value of einsum to 1
		#attack_range = np.ones(num_players) + 10
		#if self.log:
		# print(f"Mean attack range: {attack_range.mean()}")
		max_attack_range = np.max(attack_range)
		if rotation:
			XY_rot = np.einsum("mij,m...j->i...m", rot_matrix, XY)#This step leads to duplicates forming at different locations
			Z = XY_rot.copy()#for debuggin
			Z[0][Z[0] < 0] = max_attack_range+1
			XY_rot_abs = np.abs(Z)
			Y_rot_abs_passive = np.abs(XY_rot[1])#quite sure there's a faster way but go with this for now
		else:
			XY_rot_abs = np.abs(XY)

		if rotation:
			X_end = XY_rot_abs[0]
			Y_end = XY_rot_abs[1]
			Y_end_passive = Y_rot_abs_passive
		else:
			X_end = np.reshape(XY_rot_abs[..., 0], [num_players, num_players])
			Y_end = np.reshape(XY_rot_abs[..., 1], [num_players, num_players])

		XY_out = X_end + (1/prop_side)*Y_end
		XY_out_passive = X_end+Y_end_passive
		masks = np.transpose(np.transpose(XY_out, [1,0]) <= attack_range, [1,0]) | (XY_out_passive <= self.passive_range)
		masks = np.transpose(masks, [1,0])
		self.attacked[...] = 0
		self.attacked_dist[...] = 0
		self.can_see[...] = 0
		self.dead[...] = 0
		self.rewards[...] = 0#make it so that one dies
		alive_ids = np.asarray(alive)
		k = -1
		for i in range(self.sides):
			for j in range(self.players_per_side[i]):
				player = self.player_array[i][j]
				if not player.alive:
					continue
				k += 1
				"""self.attacked[i] denotes the amount of damage the side i inflicts on the other sides"""
				
				living_k = living.copy()
				living_k[living_k] = masks[k]
				if player.type == 1 and self.attack_turn % self.archer_freq != 0:
					self.attacked[i, living_k] += epsilon*player.strength
				else:
					self.attacked[i, living_k] += player.strength
		#Potenially slow bad but
		#if self.log:
		# print(self.attacked)
		self.attacked_dist = self.attacked.copy()
		self.can_see = self.attacked > 0
		#can_see contains all players each side can see
		attacked_sum = np.sum(self.attacked, axis = 0)
		for i in range(self.sides):
			self.attacked[i] = attacked_sum - self.attacked[i]
		k = 0
		for i in range(self.sides):
			for j in range(self.players_per_side[i]):
				if not self.player_array[i][j].alive:
					continue
				player_alive = self.player_array[i][j].damage(self.attacked[i][k]+self.continue_penalty)
				if not player_alive:
					self.dead[i] += 1
					self.remaining_players[i] -= 1
				k += 1
		total_death = np.sum(self.dead)
		for i in range(self.sides):
			self.rewards[i] = total_death - 2*self.dead[i]#how many more died then your side
		for i in range(self.sides):
			for j in range(self.players_per_side[i]):
				if self.player_array[i][j].alive and self.player_array[i][j].id in alive:
					alive.remove(self.player_array[i][j].id)
		for id in alive:#all dead
			self.space.remove(self.balls[id-1], self.balls[id-1].body)
			self.current_balls.remove(self.balls[id-1])


		self.set_board()
		self.attack_turn += 1
	def game_step(self):
		self.move()
		self.render_game()
		if self.t != 0 and self.t % self.moves_without_attack == 0:
			self.attack()
	def run_env(self):
		"""
		Test if environment is running properly. Show argument must be true
		"""
		self.start_game()
		if self.save_imgs:
			if not os.path.exists( self.base_directory+ "/animation"):
				os.makedirs( self.base_directory + "/animation")
			folders = os.listdir(self.base_directory  + "/animation")
		self.vel_mags = np.zeros(self.player_num)
		start_pos = None
		for t in tqdm(itertools.count()):
			try:
				for event in pygame.event.get():
					if event.type == pygame.QUIT:
						pygame.quit()
						break
					elif event.type == pygame.MOUSEBUTTONDOWN:#By clicking two times form arrow that gives force for players to move
						if event.button == 1:
							if start_pos == None:
								start_pos = pygame.mouse.get_pos()
							else:
								self.interact_move(start_pos, pygame.mouse.get_pos())
								start_pos = None
						elif event.button == 4:
							self.vec_width += self.vec_width_diff
							if self.log:
								print(f"Current vec width is {self.vec_width}")
						elif event.button == 5:
							self.vec_width -= self.vec_width_diff
							if self.log:
								print(f"Current vec width is {self.vec_width}")
						elif event.button == 3:
							self.clear_screen()
							if self.log:
								print(f"Cleared screen")
						elif event.button == 2:
							self.interact_side += 1
							self.interact_side %= self.sides
							if self.log:
								print(f"Changed side to {self.interact_side}")
					else:
						None

				self.t = t
				self.game_step()
				if self.save_imgs:
					self.show_board(folder=self.base_directory   + f"/animation/animation_players_{len(folders)//2}",save=self.save_animation, step=t, title="Moves of {}".format(self.remaining_players))
					self.show_interact_board(folder=self.base_directory   + f"/animation/animation_interact_{len(folders)//2}",save=self.save_animation, step=t, title="Moves of {}".format(self.remaining_players))

				self.reset_web()
				self.set_board()
				if t>=self.terminate_turn or self.end():
					pygame.quit()
					break

			except Exception as e:
				import traceback
				print(f"{e}")
				print(traceback.format_exc())
				pygame.quit()
				break
	def update_step(self):
		self.game_step()
		self.reset_web()
		self.set_board()
		done = self.end()
		if self.t>=self.terminate_turn or self.end():
			done = True
		
		self.t += 1
		self.finished_sides[...] = 0
		self.get_sight()
		dones = [done for _ in range(self.sides)]
		infos = [{} for _ in range(self.sides)]
		if self.is_train:
			if (self.stage != self.act_board_size) and ((self.total_moves+1) % self.stage_update_num == 0):
				print(f"stage is {self.stage}")
				self.stage *= 2
			self.total_moves += 1
		return self.obs, self.rewards, dones, infos
	def step(self, action):
		if not self.started:
			self.start_game()
		side = self.side
		action = np.reshape(action, [self.act_board_size, self.act_board_size, 2])
		action *= self.player_force_prop
		if self.is_train:
			size = self.act_board_size // self.stage
			for i in range(self.stage):
				for j in range(self.stage):
					for k in range(2):
						action_segment = action[size*i:size*(i+1), size*j:size*(j+1), k]
						action_mean = action_segment.mean()
						action_std = action_segment.std()
						action[size*i:size*(i+1), size*j:size*(j+1), k] = np.random.normal(action_mean, action_std)
		self.action[self.side] = action.copy()
		self.move_board[side] = cv2.resize(action, (self.board_size[0], self.board_size[1]))
		if self.save_imgs:
			self.show_board(folder=self.base_directory   + f"/animation/animation_players_{len(folders)//2}",save=self.save_animation, step=t, title="Moves of {}".format(self.remaining_players))
			self.show_interact_board(folder=self.base_directory   + f"/animation/animation_interact_{len(folders)//2}",save=self.save_animation, step=t, title="Moves of {}".format(self.remaining_players))
		self.finished_sides[side] = 1
		self.side += 1
		self.side %= self.sides
		if self.finished_sides[self.finished_sides == 0].shape[0] == 0:
			return self.update_step()
		return None, None, None, None
	def reset(self):
		self.__init__(**self.kwargs)
		self.start_game()
		self.vel_mags = np.zeros(self.player_num)
		self.t = 0
		self.get_sight()
		return [self.obs]
	def render(self, mode='human', close=False):
		self.render_output = self.beautiful_output.copy()
		self.arrow_output = np.zeros_like(self.render_output)
		#self.screen.blit(self.surf, (0,0))
		#draw_width = self.draw_width
		colors = self.get_n_colors()
		for i in range(self.sides):
			for i2 in range(self.sides):
				for j in range(self.players_per_side[i2]):
					player = self.player_array[i2][j]
					if not player.alive or not (self.full_view or self.can_see[i, player.id-1]):
						continue
					try:
						xys = self.board_sight[player.id-1, :4].copy()
						x = xys[:2]
						y = xys[2:]
						y += x
						x = self.switch_to_pymunk(x)
						y = self.switch_to_pymunk(y)
						cv2.circle(self.render_output[i], tuple(self.switch_to_pymunk([int(player.position[0]), int(player.position[1])])), int(player.radius) if int(player.radius) > 0 else 1, tuple([int(m) for m in colors[i2]]))
						cv2.line(self.render_output[i], tuple([int(x[0]), int(x[1])]), tuple([int(y[0]), int(y[1])]), tuple([int(m) for m in colors[i2]]))
					except Exception as e:
						print(f"{e}. color: {color}. position: {player.position}, radius: {player.radius}, alive: {player.alive}")
						import traceback

						print(traceback.format_exc())
			arrow_size = self.board_size[0]/(self.stage*2)
			size = self.act_board_size//self.stage
			for a0 in range(0,self.stage):
				for a1 in range(0,self.stage):
					arrow = self.action[i, a0*size, a1*size].copy()
					arrow *= arrow_size
					x=a0*2+1
					y=a1*2+1
					start = self.switch_to_pymunk([int(x*arrow_size), int(y*arrow_size)])
					end = self.switch_to_pymunk([int(x*arrow_size+arrow[0]), int(y*arrow_size+arrow[1])])

					cv2.arrowedLine(self.arrow_output[i], tuple(start), tuple(end), (255, 255, 255))
		self.render_output = np.concatenate([self.render_output, self.arrow_output], axis=1)
		return [self.render_output]
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
					if not player.alive or not (self.full_view or self.can_see[i, player.id-1]):
						continue
					position = player.position
					position = np.asarray(position)
					position[position > self.board_size[0]-1] =  self.board_size[0]-1
					self.obs_full[i, int(position[0]), int(position[1]), 1] = player.hp if i == i2 else 0
					self.obs_full[i, int(position[0]), int(position[1]), 2] = player.hp if i != i2 else 0
					self.obs_full[i, int(position[0]), int(position[1]), 3:5] = player.velocity.copy() if i == i2 else [0,0]
					self.obs_full[i, int(position[0]), int(position[1]), 5:7] = player.velocity.copy() if i != i2 else [0,0]
					self.obs_full[i, int(position[0]), int(position[1]), 7] = self.attacked_dist[i, player.id-1]
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
			resized_obs = cv2.resize(self.obs_full[i, ..., 1:], (self.obs_board_size, self.obs_board_size))
			self.obs[i, ..., 1:] = resized_obs.copy()
	def show_board(self, folder = "./animation", save = False,  step = 0, title= None):
		if not os.path.exists(folder):
			os.makedirs(folder)
		output = self.map.copy()
		fig, ax = plt.subplots(1)
		if title != None:
			plt.title(title)
		ax.set_aspect('equal')
		ax.set_xlim(0, self.board_size[0]-1)
		ax.set_ylim(0, self.board_size[1]-1)
		ax.imshow(output)

		colors = self.get_n_colors()
		type_colors = self.get_n_colors(self.num_types)
		for i in range(self.sides):
			color = colors[i]
			for j in range(self.players_per_side[i]):
				player = self.player_array[i][j]
				if player.alive:#do not show dead players
					position = player.position
					circ = Circle(position, radius = player.radius, color = np.asarray(color)/255., fill=False)
					#circ_border = Circle(position, radius = player.base_vision, color = np.asarray(self.colors[player.type])/255., fill=False)

					hp = player.hp
					ax.text(position[0]-0.5, position[1]-0.5, int(hp), fontsize = 9, color = np.asarray(color)/255.)

					ax.add_patch(circ)
					#ax.add_patch(circ_border)

					if self.draw_vels and player.vel != None:
						x, y = player.vel

						ax.plot(x, y, color = np.asarray(color)/255., linewidth=2.)
					if self.draw_connections:



						if self.webs[player.id-1] is not None:
							for m in range(self.webs[player.id-1].shape[0]):
								dest = position+self.webs[player.id-1][m]
								ax.plot([position[0], dest[0]], [position[1], dest[1]], color = np.asarray(color)/255., linewidth=0.5)

		if not save:
			plt.show()
			plt.close()
			return
		plt.savefig(folder +"/"+ str(step)+".jpg")
		plt.close()
	def show_interact_board(self, folder = "./animation", save = False,  step = 0, title= None):
		if not os.path.exists(folder):
			os.makedirs(folder)
		fig, ax = plt.subplots(1)
		if title != None:
			plt.title(title)
		ax.set_aspect('equal')
		ax.set_xlim(0, self.num_arrows)
		ax.set_ylim(0, self.num_arrows)
		output_xy = self.move_board[self.interact_side].copy()*self.arrow_scale_constant
		output_xy[output_xy == 0] = 0.0000001
		jump_step = self.board_size[0]//self.num_arrows
		output_xy = output_xy[::jump_step, ::jump_step]
		ax.quiver(output_xy[..., 0], output_xy[..., 1])

		if not save:
			plt.show()
			plt.close()
			return
		plt.savefig(folder +"/"+ str(step)+".jpg")
		plt.close()
	def show_rank_board(self, folder = "./animation_rank", save = False, step = 0, title= None):
		if not os.path.exists(folder):
			os.makedirs(folder)
		colors = self.get_n_colors()
		output = np.zeros([self.board_size[1], self.board_size[0]])
		plt.xlim(0, self.board_size[0]-1)
		plt.ylim(0, self.board_size[1]-1)
		if title != None:
			plt.title(title)
		plt.imshow(output)
		for i in range(self.sides):
			color = colors[i]
			for j in range(self.players_per_side[i]):
				if self.player_array[i][j].alive:#do not show dead players
					position = self.player_array[i][j].position
					rank = self.player_array[i][j].rank
					plt.text(position[0], position[1], int(rank), fontsize = 9, color = np.asarray(color)/255.)
		if not save:
			plt.show()
			plt.close()
			return
		plt.savefig(folder +"/"+ str(step)+".jpg")
		plt.close()
	def end(self):
		#if np.max(self.remaining_players) == np.sum(self.remaining_players):
		#   return True#only one side remains
		if np.sum(self.remaining_players) < np.sum(self.players_per_side)*self.min_frac or np.max(self.remaining_players) == np.sum(self.remaining_players):#If the total number is half,
			return True
		for i in range(self.sides):
			if self.remaining_players[i] < self.min_players:
				return True
		return False
