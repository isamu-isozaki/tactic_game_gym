import numpy as np
import random
import math

class Player:
	def __init__(self, strength, hp, k, id, side, **kwargs):
		#The strength and smartness take integer values between 0 and 100. The position is the inital position of the player
		self.strength = strength
		self.hp = hp
		self.k = k#spring constant
		self.id = id
		self.aptitude = self.strength+self.hp
		self.side = side
		self.type=-1
		self.mass = 1
		self.position = np.zeros(2, dtype=np.float16)
		self.alive = True
		self.force = None
		self.r_a = None
		self.player_force = None
		self.max_speed = None
		self.speed = None
		self.radius = None
		self.base_vision = None
		self.rank = 1
		self.superior_id = None
		self.superior_j = None
		self.web = []
		self.superior_pos = None
		self.sub_ids = None#id of subordinates
		self.sub_js = None
		self.sub_pos = None
		self.kill = 0
		self.reward = 0
		self.sight = None
		self.force_prop = None
		self.k=1.
		self.velocity = np.array([0,0], dtype=np.float16)
		self.vel = None
		self.angle = [0,0]
		self.height = 0
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