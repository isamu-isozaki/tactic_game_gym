from tactic_game_gym.tactic_game.env_base import Base_Env
from tactic_game_gym.tactic_game.game_args import game_args_parser

"""
Things to change:
1. Change field to velocity field
2. Change so that force is applied to stop and stop for a duration(change direction)
and start moving in new direction
3. Formula coef*(v-v_current)=force no max velocity but max force
-> Problem with above is that the acceleration needs to be close to 0 when at v
v-v_current
actually, how about initial force = that and collision force just gets applied later? than it will kinda make sense
"""
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