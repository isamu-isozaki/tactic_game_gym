from tactic_game_gym.tactic_game.player import Player

class Wall(Player):
    def __init__(self, *argv, **kwargs):
        Player.__init__(self, *argv, **kwargs)
        self.player_name = "wall"
        self.type = 3
        self.set_properties()