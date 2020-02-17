import gym
from abc import abstractclassmethod

class Base_Env(gym.Env):
	def __init__(self):
		super(Base_Env, self).__init__()
	@abstractclassmethod
	def step(self, action):
		pass
	@abstractclassmethod
	def reset(self):
		pass
	@abstractclassmethod
	def render(self, mode='human', close=False):
		pass
	