import argparse
import numpy as np

def arg_parser():#credits to OpenAI!
	"""
	Create an empty argparse.ArgumentParser.
	"""
	import argparse
	return argparse.ArgumentParser(
		description='Setting up tactic game',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
def game_args_parser():
	parser = arg_parser()
	parser.add_argument('--save_dir', type=str, default="model", help='The save directory where the checkpoint is')
	parser.add_argument('--use_base_dir',default=True, action='store_false', help='Use the current base directory')
	parser.add_argument('--strength', type=float, default=0.02, help='Mean strength of player')
	parser.add_argument('--hp', type=float, default=10.0, help='The mean HP of player')
	parser.add_argument('--std_k', type=float, default=2.0, help='Std of spring constant. Calculated by a normal distribution in sigmoid. Mean of normal is 0')
	parser.add_argument('--rand_params', default=True, action='store_false', help='Whether to randomize parameters or not')
	parser.add_argument('--rand_prop', type=float, default=0.2, help='The proportion std of the normal distribution.')
	parser.add_argument('--cap_prop', type=float, default=0.2, help='The propotion of the std of the normal distribution for capabilities(hp, strengt)')
	parser.add_argument('--max_players', type=int, default=1000, help='Maximum number of players per game')
	parser.add_argument('--sides', type=int, default=2, help='The number of sides in the war')
	parser.add_argument('--board_size', type=int, default=129, help='The number of sides in the war. Due to diamond square algorithm, must be in the form of 2^n+1')
	parser.add_argument('--scale_fac', type=int, default=8, help='Scale down ratio for the board size')
	parser.add_argument('--num_games', type=int, default=128, help='Number of games to play on parallel. This is something like batch size')
	parser.add_argument('--moves_without_attack', type=int, default=3, help='The number of moves without needing to attack')
	parser.add_argument('--num_subs', type=int, default=5, help='The maximum amount of subordinates per superior')
	parser.add_argument('--random_action', default=True, action='store_false', help='Whether the actions of the agents are random or not')
	parser.add_argument('--base_vision', type=float, default=10, help='Mean of base vision.')#This is the same as attack range
	parser.add_argument('--archer_constant', type=float, default=2.0, help='The archer constant. The attack range is n times larger but the HP is n**2 times less')
	parser.add_argument('--archer_freq', type=int, default=2, help='The frequency at which archers attack')
	parser.add_argument('--mass', type=float, default=1.0, help='Mass of players')#This is the same as attack range
	parser.add_argument('--atomic_radius', type=float, dest='r_a', default=3.0, help="""Semi atomic radius.
	The true atomic radius is calculated by r+r_a""")
	parser.add_argument('--order_constant', type=float, default=1.0, help='The max magnitude of order force')
	parser.add_argument('--player_force', type=float, default=3.0, help='The max magnitude of player force')
	parser.add_argument('--max_speed', type=float, default=6.0, help='The max speed')
	parser.add_argument('--range_factor', type=float, default=2.0, help='The factor by which to multiply the range by')
	parser.add_argument('--death_penalty', type=float, default=-0.5, help='The penalty for death per player')
	parser.add_argument('--losing_penalty', type=float, default=-0.25, help="""The penalty for losing/surrendering.
	The players will subsequently erased from the board.""")
	parser.add_argument('--defecting_penalty', type=float, default=-0.25, help='The penalty for switching sides')
	parser.add_argument('--continue_penalty', type=float, default=-0.001, help='The penalty for prolonging the game this is deducted from energy bar')
	parser.add_argument('--kill_reward', type=float, default=0.5, help='The reward for killing a rank 1 player. This will increase per rank')
	parser.add_argument('--num_types', type=int, default=3, help='The number of player types')
	parser.add_argument('--rand_troop_prop', type=float, default=0.1, help='The std of normal distribution of 1 by which to multiply the troop props')
	parser.add_argument('--cavarly_prop', type=float, default=0.15, help='The proportion of cavarly in troops.')
	parser.add_argument('--archer_prop', type=float, default=0.2, help='The proportion of archers in troops')
	parser.add_argument('--infantry_prop', type=float, default=0.65, help='The proportion of infantry in troops')
	parser.add_argument('--cavarly_scale', type=float, default=1.2, help='How much larger the radius of the cavarly is than the infantry')
	parser.add_argument('--cavarly_hp', type=float, default=1.2, help='How much more hp cavarly has than the infantry. This is to account for their larger size.')
	parser.add_argument('--cavarly_force', type=float, default=1.5*1.2**3, help='How much more force cavarly has than the infantry. This is to account for their larger mass.')
	parser.add_argument('--cavarly_max_speed', type=float, default=1.5, help='How much more larger the max speed of the cavarly is')
	parser.add_argument('--cavarly_k', type=float, default=1.2**3, help='The multiplier to the spring coefficient for cavarly')


	"""
	Change below according to experiments
	"""

	parser.add_argument('--player_force_prop', type=float, default=3.0, help='The coefficient of the player force')
	parser.add_argument('--drag_force_prop', type=float, default=4.5, help='The coefficient of the drag force')
	parser.add_argument('--spring_force_prop', type=float, default=0.25, help='The coefficient of the spring force')
	parser.add_argument('--g', type=float, default=20.0, help='The gravitational constant')
	parser.add_argument('--max_angle', type=float, default=np.pi/3, help='The maximum angle possible for the terrain to have')


	parser.add_argument('--game_timestep', type=float, default=.2, help='The time step between each step')

	parser.add_argument('--test_env', default=False, action='store_true', help='Test if environment is running properly')
	parser.add_argument('--attrange_div_const', type=float, default=1.0, help='The proportion of max_speed by which to divide velocity magnitude by')


	parser.add_argument('--test_mobilize', default=False, action='store_true', help='Test if mobilization is running properly')

	parser.add_argument('--terminate_turn', type=int, default=10000, help='The amount of time steps until stopping environment')
	parser.add_argument('--save_animation', default=True, action='store_false', help='The amount of time steps until stopping environment')

	parser.add_argument('--prop_side', type=float, default=0.5, help='The proportion of attack that goes behind')


	parser.add_argument('--draw_width', type=float, default=2., help='The width of lines and circles for the pygame objects')
	parser.add_argument('--vec_width_diff', type=int, default=15, help='The amount by which the vec width changes when pressing up and down on the arrow keys')

	parser.add_argument('--vec_steps', type=int, default=10, help='The number of steps/resolution of each vector\'s area of effect')
	parser.add_argument('--vec_mag_div_constant_frac', type=float, default=1, help='The proportion of the board_size to divide the arrow magnitude by')
	parser.add_argument('--num_arrows', type=int, default=20, help='The number of arrows shown on interact board')
	parser.add_argument('--arrow_scale_constant', type=int, default=2000, help='The scale by which to increase the arrow\'s magnitude')


	parser.add_argument('--save_imgs', default=False, action='store_true', help='Whether or not save images')

	parser.add_argument('--draw_connections', default=False, action='store_true', help='Whether or not to draw lines for matplotlib')
	parser.add_argument('--draw_vels', default=False, action='store_true', help='Whether or not to draw lines for matplotlib')

	parser.add_argument('--test', default=False, action='store_true', help='Whether to test model or not')
	parser.add_argument('--passive_range', type=float,default=3, help='The minimum attack range a player can have')

	parser.add_argument('--obs_board_size', type=int,default=129, help='The board size for observation board')
	parser.add_argument('--act_board_size', type=int,default=17, help='The board size for action board')
	parser.add_argument('--map_board_size', type=int,default=65, help='The initial boardsize which will get resized')

	parser.add_argument('--full_view', default=False, action='store_true', help='Whether to see the whole view or not')
	parser.add_argument('--show', default=False, action='store_true', help='Whether to display the game/whether to use pygame or not')
	parser.add_argument('--log', default=False, action='store_true', help='Whether to display the log not')
	parser.add_argument('--attack_div_frac', default=0.5, type=float, help='The proportion of the maximum attack to normalize for')
	return parser
