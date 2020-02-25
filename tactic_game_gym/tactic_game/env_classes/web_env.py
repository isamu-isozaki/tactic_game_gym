from tactic_game_gym.tactic_game.env_classes.game_base_env import Get_Sight
import numpy as np

class Setup_Web(Get_Sight):
    def __init__(self, **kwargs):
        Get_Sight.__init__(self, **kwargs)
    def reset_web(self):
        self.webs = []
        for _ in range(self.player_num):
            self.webs.append(None)
        self.vel_mags[...] = 0
    def get_web(self, player):
        web = None
        if self.webs[player.id] is not None:
            web = self.webs[player.id]
        else:
            player_pos = self.board_sight[player.id, :2].copy()
            web = np.reshape(self.board_sight[player.web, :2].copy(), (-1, 2))
            if player.superior_id == None and player.sub_ids == None:
                return None
            alive = self.return_alive(player.web)
            web = web[alive==1]

            if web is None or web.shape[0] == 0:
                web = None

            if web is not None:
                web -= player_pos
            self.webs[player.id] = web
        return web
    def get_web_and_mag(self, player):
        web = None
        if self.webs[player.id-1] is not None:
            web = self.webs[player.id-1]
        else:
            player_pos = self.board_sight[player.id, :2].copy()
            web = np.reshape(self.board_sight[player.web, :2].copy(), (-1, 2))
            if player.superior_id == None and player.sub_ids == None:
                return None, None

            alive = self.return_alive(player.web)
            web = web[alive==1]
            if web is None or web.shape[0] == 0:
                web = None

            if web is not None:
                web -= player_pos
            self.webs[player.id -1] = web
        mag = None
        if web is None:
            return web, mag
        if self.mags[player.id-1] is not None:
            mag = self.mags[player.id-1]
        else:
            mag = np.linalg.norm(web, axis=1)
            self.mags[player.id -1] = mag
        return web, mag	