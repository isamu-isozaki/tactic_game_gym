from tactic_game_gym.tactic_game.env_classes.swarm_intelligence_env import Setup_Swarm_Intelligence
from tactic_game_gym.tactic_game.env_classes.springs_env import Setup_Springs
import numpy as np

class Setup_Drag_Force(Setup_Swarm_Intelligence, Setup_Springs):
    def __init__(self, **kwargs):
        Setup_Springs.__init__(self, **kwargs)
        Setup_Swarm_Intelligence.__init__(self, **kwargs)
    def get_drag(self, player, force):
        if self.get_height(*player.position) != 0:
            return np.asarray([0,0], dtype=np.float16)
        above_limit = np.abs(force[np.abs(force) > self.drag_force_prop])
        if above_limit.shape[0] is not 0:
            force /= above_limit.max()
            force *= self.drag_force_prop
        return force

class Setup_Rotate_Force(Setup_Swarm_Intelligence, Setup_Springs):
    def __init__(self, **kwargs):
        Setup_Springs.__init__(self, **kwargs)
        Setup_Swarm_Intelligence.__init__(self, **kwargs)
    def rotate_force(self, player, force, z):
        """
        Goal of this function:
        rotate the force and deduct/add force taken from gravity so that it makes it harder to climb up mountains and can speed up when going down mountains
        Current force-> 2d
        Convert force to -> 3d
        x -> x cos(angle)

        y -> y cos(angle)

        z -> root(x^2+y^2) sin(angle)

        Get unit vector by dividing by sqrt(x^2+y^2)

        cos alpha with the gravity will be the negative z component of the 3d vector

        multiply cos alpha by the mg scalar and multiply by the unit vector
        """
        position = player.position
        position[position > self.board_size[0]-1] =  self.board_size[0]-1
        position = position.tolist()
        force_mag = 0
        if force[0] != 0 or force[1] != 0:
            force_mag = np.linalg.norm(force)
        force_angles = None
        if force_mag == 0:
            force_angles = np.ones([2])/np.sqrt(2)
        else:
            force_angles = force/force_mag
        try:
            
            force_3d_unit = np.asarray([force_angles[0]*player.cos,\
             force_angles[1]*player.cos,\
            z], dtype=np.float16)
        except Exception as e:
            if self.log:
                print(f"{e}. force angles: {force_angles}, force: {force}")
        cos_weight = -force_3d_unit[2]
        cos_weight *= player.mass*self.g
        try:
            force_3d = force_3d_unit*(force_mag+cos_weight)
        except Exception as e:
            print(f"{e}. force_3d_unit: {force_3d_unit}. force_mag: {force_mag}. cos_weight: {cos_weight}")
        return force_3d[:2]
