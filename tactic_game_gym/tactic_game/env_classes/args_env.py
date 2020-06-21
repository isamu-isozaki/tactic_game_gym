from tactic_game_gym.tactic_game.env_base import Base_Env
from tactic_game_gym.tactic_game.game_args import game_args_parser


class Args_Env(Base_Env):
    def __init__(self, **kwargs):
        #Thanks https://stackoverflow.com/questions/5624912/kwargs-parsing-best-practice
        arg_parser = game_args_parser()
        args, _ = arg_parser.parse_known_args()
        self.kwargs = vars(args)
        self.kwargs.update(kwargs)
        for k,v in self.kwargs.items():
            setattr(self, k, v)
        super(Args_Env, self).__init__()
        # Set proportions accurately
        total_prop = self.wall_prop+self.infantry_prop+self.archer_prop+self.cavarly_prop
        self.wall_prop /= total_prop
        self.infantry_prop /= total_prop
        self.archer_prop /= total_prop
        self.cavarly_prop /= total_prop