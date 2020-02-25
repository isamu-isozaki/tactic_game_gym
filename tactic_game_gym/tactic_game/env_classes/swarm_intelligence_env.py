from tactic_game_gym.tactic_game.env_classes.game_base_env import Get_Sight
import numpy as np

class Setup_Swarm_Intelligence(Get_Sight):
    def __init__(self, **kwargs):
        Get_Sight.__init__(self, **kwargs)
        self.align_dest = np.zeros([self.player_num, 2])
        self.align_mags = np.zeros(self.player_num)
        self.cohesion_dest = np.zeros([self.player_num, 2])
        self.cohesion_mags = np.zeros(self.player_num)


    def get_boids(self, living):
        #Remove separation because collision already does that.
        #Modified algorithm so that it goes towards the web of the player and ignores all other factors
        return self.cohesion(living)
        
    def cohesion(self, living, epsilon=10**-10):
        self.cohesion_dest[:] = 0
        self.cohesion_mags[:] = 0
        apply_mag = living.copy()
        for i in range(self.sides):
            for j in range(self.players_per_side[i]):
                player = self.player_array[i][j]
                if player.alive:
                    offset = np.mean(self.board_sight[player.web, :2], axis=0)
                    offset -= player.position
                    #if (np.sum(offset)+epsilon) < (2*player.r_a+epsilon):
                    #    apply_mag[player.id] = False
                    #    offset = [0, 0]
                    try:
                        self.cohesion_dest[player.id] = offset*self.cohesion_force_prop/self.board_size[0] -player.velocity*0.1
                    except Exception as e:
                        print(f"{e}. velocity: {player.velocity}")
        return self.cohesion_dest[living]


    def alignment(self, living, epsilon=10**-10):
        self.align_dest[:] = 0
        self.align_mags[:] = 0
        for i in range(self.sides):
            for j in range(self.players_per_side[i]):
                player = self.player_array[i][j]
                if player.alive:
                    offset_vel = np.mean(self.board_sight[player.web, 2:4], axis=0)
                    self.align_dest[player.id] = offset_vel-player.velocity
        self.align_mags[living] = np.linalg.norm(self.align_dest[living], axis=1)
        return np.transpose(np.transpose(self.align_dest[living]*self.align_force_prop)/(self.align_mags[living]+epsilon))
