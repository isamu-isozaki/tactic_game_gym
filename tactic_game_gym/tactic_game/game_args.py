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
	parser.add_argument('--act_board_size', type=int,default=16, help='The board size for action board. Must be power of 2')
	parser.add_argument('--align_force_prop', type=float,default=1.0, help='The force to align the players')
	parser.add_argument('--archer_base_vision', type=float, default=20.0, help='The attack range of archer')
	parser.add_argument('--archer_density', type=float, default=1, help='The mass density of archer')
	parser.add_argument('--archer_force', type=float, default=1, help='The force coefficient of archer')
	parser.add_argument('--archer_freq', type=int, default=8, help='The frequency at which archers attack')
	parser.add_argument('--archer_hp', type=float, default=0.5, help='How much more hp archer has than the infantry. This is to account for their larger size.')
	parser.add_argument('--archer_k', type=float, default=1, help='The multiplier to the spring coefficient for archer')
	parser.add_argument('--archer_max_speed', type=float, default=1, help='How much more larger the max speed of the archer is')
	parser.add_argument('--archer_prop', type=float, default=0.3, help='The proportion of archer in troops.')
	parser.add_argument('--archer_scale', type=float, default=1, help='How much larger the radius of the archer is than the infantry')
	parser.add_argument('--arrow_scale_constant', type=int, default=2000, help='The scale by which to increase the arrow\'s magnitude')
	parser.add_argument('--atomic_radius', type=float, dest='r_a', default=6.0, help="""Semi atomic radius.
	The true atomic radius is calculated by r+r_a""")
	parser.add_argument('--attack_div_frac', default=0.5, type=float, help='The proportion of the maximum attack to normalize for')
	parser.add_argument('--attrange_div_const', type=float, default=1.0, help='The proportion of max_speed by which to divide velocity magnitude by')
	parser.add_argument('--base_vision', type=float, default=20, help='Mean of base vision.')#This is the same as attack range
	parser.add_argument('--boid_damping_factor', type=float, default=0.5, help='The coefficient of velocity used for damping for boid')
	parser.add_argument('--board_size', type=int, default=257, help='The number of sides in the war. Due to diamond square algorithm, must be in the form of 2^n+1')
	parser.add_argument('--cap_prop', type=float, default=0.8, help='The propotion of the std of the normal distribution for capabilities(hp, strengt)')
	parser.add_argument('--cavarly_base_vision', type=float, default=10, help='The attack range of cavarly')
	parser.add_argument('--cavarly_density', type=float, default=1, help='The mass density of cavarly')
	parser.add_argument('--cavarly_force', type=float, default=2*1.5**3, help='How much more force cavarly has than the infantry. This is to account for their larger mass.')
	parser.add_argument('--cavarly_hp', type=float, default=1.5, help='How much more hp cavarly has than the infantry. This is to account for their larger size.')
	parser.add_argument('--cavarly_k', type=float, default=1.5**3, help='The multiplier to the spring coefficient for cavarly')
	parser.add_argument('--cavarly_max_speed', type=float, default=1.5, help='How much more larger the max speed of the cavarly is')
	parser.add_argument('--cavarly_prop', type=float, default=0.1, help='The proportion of cavarly in troops.')
	parser.add_argument('--cavarly_scale', type=float, default=1.5, help='How much larger the radius of the cavarly is than the infantry')
	parser.add_argument('--cohesion_damping', type=float, default=50.0, help='The fraction to multiply offset by')
	parser.add_argument('--cohesion_force_prop', type=float, default=24.0, help='The force that makes the players move in the same direction')
	parser.add_argument('--cohesion_force_prop_after_damping', type=float, default=3.0, help='The coefficient to apply after applying damping')
	parser.add_argument('--continue_penalty', type=float, default=-0.003, help='The penalty for prolonging the game this is deducted from energy bar')
	parser.add_argument('--damage_reward_frac', type=float, default=0.1, help='The coefficient to hard coded reward of damage')
	parser.add_argument('--death_penalty', type=float, default=-0.5, help='The penalty for death per player')
	parser.add_argument('--drag_force_prop', type=float, default=4.5, help='The coefficient of the drag force')
	parser.add_argument('--draw_connections', default=False, action='store_true', help='Whether or not to draw lines for matplotlib')
	parser.add_argument('--draw_vels', default=False, action='store_true', help='Whether or not to draw lines for matplotlib')
	parser.add_argument('--draw_width', type=float, default=2., help='The width of lines and circles for the pygame objects')
	parser.add_argument('--ended_moves', default=0, type=int, help='The number of moves where training ended')
	parser.add_argument('--full_view', default=False, action='store_true', help='Whether to see the whole view or not')
	parser.add_argument('--g', type=float, default=0.5, help='The gravitational constant')
	parser.add_argument('--game_timestep', type=float, default=.2, help='The time step between each step')
	parser.add_argument('--hp', type=float, default=10, help='The mean HP of player')
	parser.add_argument('--infantry_base_vision', type=float, default=10., help='The attack range of infantry.')
	parser.add_argument('--infantry_density', type=float, default=1., help='The mass density of infantry.')
	parser.add_argument('--infantry_force', type=float, default=1, help='The force coefficient of infantry')
	parser.add_argument('--infantry_hp', type=float, default=1, help='How much more hp cavarly has than the infantry. This is to account for their larger size.')
	parser.add_argument('--infantry_k', type=float, default=1, help='The multiplier to the spring coefficient for cavarly')
	parser.add_argument('--infantry_max_speed', type=float, default=1, help='How much more larger the max speed of the cavarly is')
	parser.add_argument('--infantry_prop', type=float, default=0.4, help='The proportion of cavarly in troops.')
	parser.add_argument('--infantry_scale', type=float, default=1, help='How much larger the radius of the cavarly is than the infantry')
	parser.add_argument('--init_stage', default=1, type=int, help='The initial stage of action stage')
	parser.add_argument('--is_train', default=False, action='store_true', help='Is training')
	parser.add_argument('--kill_reward', type=float, default=0.5, help='The reward for killing a rank 1 player. This will increase per rank')
	parser.add_argument('--log', default=False, action='store_true', help='Whether to display the log not')
	parser.add_argument('--map_board_size', type=int,default=65, help='The initial boardsize which will get resized')
	parser.add_argument('--mass', type=float, default=1.0, help='Mass of players')#This is the same as attack range
	parser.add_argument('--max_angle', type=float, default=np.pi/3, help='The maximum angle possible for the terrain to have')
	parser.add_argument('--max_players', type=int, default=1000, help='Maximum number of players per game')
	parser.add_argument('--max_speed', type=float, default=6.0, help='The max speed')
	parser.add_argument('--min_frac', default=1/8., type=float, help='The proportion of players at which the game ends')
	parser.add_argument('--min_players', default=50, type=int, help='The minimum number of players at which point the game ends')
	parser.add_argument('--moves_without_attack', type=int, default=3, help='The number of moves without needing to attack')
	parser.add_argument('--moves_without_model', default=20, type=int, help='The number of moves until getting the next the step from model')
	parser.add_argument('--num_arrows', type=int, default=20, help='The number of arrows shown on interact board')
	parser.add_argument('--num_subs', type=int, default=4, help='The maximum amount of subordinates per superior')
	parser.add_argument('--num_types', type=int, default=4, help='The number of player types')
	parser.add_argument('--obs_board_size', type=int,default=65, help='The board size for observation board')
	parser.add_argument('--passive_range', type=float,default=3, help='The minimum attack range a player can have')
	parser.add_argument('--penalty_discount', type=float,default=0.3, help='The coefficient of the penalty if the reward is negative.')
	parser.add_argument('--player_force', type=float, default=1.0, help='The max magnitude of player force')
	parser.add_argument('--player_force_prop', type=float, default=3.0, help='The coefficient of the player force')
	parser.add_argument('--profile', default=False, action='store_true', help='Whether to profile or not')
	parser.add_argument('--prop_side', type=float, default=0.5, help='The proportion of attack that goes behind')
	parser.add_argument('--rand_params', default=True, action='store_false', help='Whether to randomize parameters or not')
	parser.add_argument('--rand_prop', type=float, default=0.2, help='The proportion std of the normal distribution.')
	parser.add_argument('--rand_troop_prop', type=float, default=0.1, help='The std of normal distribution of 1 by which to multiply the troop props')
	parser.add_argument('--range_factor', type=float, default=1, help='The factor by which to multiply the range by')
	parser.add_argument('--replsion_force_prop', type=float, default=0.1, help='The force of repulsion between playes')
	parser.add_argument('--save_animation', default=True, action='store_false', help='The amount of time steps until stopping environment')
	parser.add_argument('--save_dir', type=str, default="model", help='The save directory where the checkpoint is')
	parser.add_argument('--save_imgs', default=False, action='store_true', help='Whether or not save images')
	parser.add_argument('--scale_fac', type=int, default=8, help='Scale down ratio for the board size')
	parser.add_argument('--seen_reward_frac', type=int, default=0.01, help='The coefficient to hard coded reward for seeing')
	parser.add_argument('--show', default=False, action='store_true', help='Whether to display the game/whether to use pygame or not')
	parser.add_argument('--sides', type=int, default=2, help='The number of sides in the war')
	parser.add_argument('--spring_damping_factor', type=float, default=1, help='The coefficient of velocity used for damping for spring')
	parser.add_argument('--spring_force', type=float, default=3.0, help='The coefficient of the spring force')
	parser.add_argument('--spring_force_prop', type=float, default=1, help='The coefficient of the spring force')
	parser.add_argument('--stage_update_num', default=3e5, type=float, help='The number of moves until getting the next the step from model')
	parser.add_argument('--std_k', type=float, default=2.0, help='Std of spring constant. Calculated by a normal distribution in sigmoid. Mean of normal is 0')
	parser.add_argument('--stop_slide', default=True, action='store_false', help='Stop sliding')
	parser.add_argument('--strength', type=float, default=0.01, help='Mean strength of player')
	parser.add_argument('--terminate_turn', type=int, default=10000, help='The amount of time steps until stopping environment')
	parser.add_argument('--test', default=False, action='store_true', help='Whether to test model or not')
	parser.add_argument('--test_env', default=False, action='store_true', help='Test if environment is running properly')
	parser.add_argument('--test_mobilize', default=False, action='store_true', help='Test if mobilization is running properly')
	parser.add_argument('--use_base_dir',default=True, action='store_false', help='Use the current base directory')
	parser.add_argument('--use_arrow',default=False, action='store_true', help='Use arrow when displaying render')
	parser.add_argument('--use_boid',default=False, action='store_true', help='Use boids for gathering the players together')
	parser.add_argument('--use_spring',default=False, action='store_true', help='Use spring for gathering the players together')
	parser.add_argument('--vec_mag_div_constant_frac', type=float, default=1, help='The proportion of the board_size to divide the arrow magnitude by')
	parser.add_argument('--vec_steps', type=int, default=10, help='The number of steps/resolution of each vector\'s area of effect')
	parser.add_argument('--vec_width_diff', type=int, default=15, help='The amount by which the vec width changes when pressing up and down on the arrow keys')
	parser.add_argument('--wall_base_vision', type=float, default=40, help='The base vision of walls')
	parser.add_argument('--wall_density', type=float, default=20, help='The density of walls')
	parser.add_argument('--wall_force', type=float, default=20, help='How much more force cavarly has than the infantry. This is to account for their larger mass.')
	parser.add_argument('--wall_hp', type=float, default=20, help='How much more hp wall has than the infantry. This is to account for their larger size.')
	parser.add_argument('--wall_k', type=float, default=80, help='The multiplier to the spring coefficient for cavarly')
	parser.add_argument('--wall_max_speed', type=float, default=0.1, help='How much more larger the max speed of the cavarly is')
	parser.add_argument('--wall_prop', type=float, default=0.1, help='The proportion of walls in troops.')
	parser.add_argument('--wall_scale', type=float, default=2, help='How much larger the radius of the cavarly is than the infantry')
	parser.add_argument('--win_reward', type=float, default=100, help='The reward for winning the game')
	return parser