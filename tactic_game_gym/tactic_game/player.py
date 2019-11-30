import numpy as np
import random
import math
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import pygame
from pygame.locals import *
import pymunk
np.seterr(all='raise')
"""
Changes from v2.1 to v3
No attack angles
"""
class Player:
	def __init__(self, strength, hp, k, id, side):
		#The strength and smartness take integer values between 0 and 100. The position is the inital position of the player
		self.strength = strength
		self.hp = hp
		self.k = k#spring constant
		self.id = id
		self.aptitude = self.strength+self.hp
		self.side = side
	type=-1
	mass = 1
	position = np.zeros(2)
	movement = np.zeros(2)
	alive = True
	force = None
	r_a = None
	player_force = None
	max_speed = None
	speed = None
	radius = None
	base_vision = None
	rank = 1
	superior_id = None
	superior_j = None
	superior_pos = None
	sub_ids = None#id of subordinates
	sub_js = None
	sub_pos = None
	kill = 0
	reward = 0
	sight = None
	force_prop = None
	k=1.
	velocity = [0,0]
	vel = None
	params = np.zeros(6)#Holds Position, Rank,  Side, Strength, hp
	angle = [0,0]
	height = 0
	def set_params(self):
		try:
			self.params = np.asarray([*self.position, *self.velocity, self.rank, self.side, self.strength, self.hp])
		except Exception as e:
			print(f"{e}. params: {params}")

	"""
	Mechanics Change:
	1. Set destinations for all players and move there position to that location.
	2. If player is outside of bounds, push the x or y axis of player until player is within bounds again.
	3. If the players are closer than 2, push back on both sides by a vector connecting the center of the circles by (2-d)/2 to either side
	4. Loop until all players are within bounds and away from each other by 2
	"""
	def apply_force(self, force, body):
		try:
			self.force = force.tolist()
			body.body.apply_force_at_local_point(self.force, (0, 0))
		except Exception as e:
			print(f"{e}. force:{force}")


	def damage(self, damage):
		if not self.alive:
			return False
		self.hp -= damage
		if self.hp < 0:
			self.alive = False
			return False
		return True
	def show_sight(self, board_size, sides):
		board_sight = np.zeros([board_size[1], board_size[0]])#Thanks https://stackoverflow.com/questions/34902477/drawing-circles-on-image-with-matplotlib-and-numpy
		fig, ax = plt.subplots(1)
		ax.set_aspect('equal')
		ax.imshow(board_sight)
		for i in range(self.sight.shape[0]):
			rgb = self.sight[i][3]/float(sides)
			circ = Circle(self.sight[i][:2], radius = 1, color = [rgb, 1, 1])
			ax.add_patch(circ)
		plt.show()
		plt.close()
