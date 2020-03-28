from tactic_game_gym.tactic_game.player import Player

class Cavarly(Player):
    def __init__(self, *argv, **kwargs):
        Player.__init__(self, *argv, **kwargs)
        self.player_name = "cavarly"
        self.type = 1
        self.set_properties()