from tactic_game_gym.tactic_game.player import Player

class Infantry(Player):
    def __init__(self, *argv, **kwargs):
        Player.__init__(self, *argv, **kwargs)
        self.player_name = "infantry"
        self.type = 2
        self.set_properties()