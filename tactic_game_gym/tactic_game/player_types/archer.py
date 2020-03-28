from tactic_game_gym.tactic_game.player import Player

class Archer(Player):
    def __init__(self, *argv, **kwargs):
        Player.__init__(self, *argv,  **kwargs)
        self.player_name = "archer"
        self.type = 0
        self.set_properties()