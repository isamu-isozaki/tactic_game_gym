from tactic_game_gym.envs.game_args import game_args_parser

from tactic_game_gym.envs.game_env import Game_Env_v0 as Game_Env
import numpy as np
import sys

#player has two modes. Random attack mode and distributed attack mode
def main(args):
	arg_parser = game_args_parser()
	args, _ = arg_parser.parse_known_args(args)#the openai way
	#convert args to dictionary
	args = vars(args)

	env = Game_Env(**args)
	print("Finished initialization")
	if args["test_env"]:
		env.run_env()
	elif args["test_mobilize"]:
		env.mobilize()
if __name__ == "__main__":
	main(sys.argv)
