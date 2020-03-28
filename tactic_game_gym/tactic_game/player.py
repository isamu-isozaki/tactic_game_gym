import numpy as np
import random
import math

class Player:
	def __init__(self, player_strength, player_hp, player_k, player_id, player_side, **kwargs):
		#The strength and smartness take integer values between 0 and 100. The position is the inital position of the player
		self.strength = player_strength
		self.hp = player_hp
		self.k = player_k#spring constant
		self.id = player_id
		self.aptitude = self.strength+self.hp
		self.side = player_side
		self.type=-1
		self.mass = 1
		self.position = np.zeros(2, dtype=np.float16)
		self.alive = True
		self.player_name = ""
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
		self.density = 1
		self.args = kwargs
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
	def set_properties(self):
		self.radius = 1
		self.r_a = self.args["r_a"]
		self.force_prop = 1
		self.base_vision = self.args[self.player_name+"_base_vision"]+self.args[self.player_name+"_scale"]-1
		self.hp *= self.args[self.player_name+"_hp"]
		self.radius *= self.args[self.player_name+"_scale"]
		self.r_a *= self.args[self.player_name+"_scale"]
		self.player_force = self.args[self.player_name+"_force"]*self.args["player_force"]
		self.force_prop = self.args[self.player_name+"_force"]
		self.max_speed = self.args[self.player_name+"_max_speed"]*self.args["max_speed"]
		self.k *= self.args[self.player_name+"_k"]
        