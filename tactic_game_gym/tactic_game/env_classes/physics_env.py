from tactic_game_gym.tactic_game.env_classes.game_base_env import Get_Sight
import numpy as np

class Setup_Web(Get_Sight):
    def __init__(self, **kwargs):
        Get_Sight.__init__(self, **kwargs)
    def reset_web(self):
        self.webs = []
        for _ in range(self.player_num):
            self.webs.append(None)
        self.mags = []
        for _ in range(self.player_num):
            self.mags.append(None)
        self.vel_mags[...] = 0
    def get_web_and_mag(self, player):
        web = None
        if self.webs[player.id-1] is not None:
            web = self.webs[player.id-1]
        else:
            player_pos = self.board_sight[player.id-1, :2].copy()
            superior_pos = np.reshape(self.board_sight[player.superior_id-1, :2].copy(), (-1, 2)) if player.superior_id != None else None
            sub_pos = np.reshape(self.board_sight[np.asarray(player.sub_ids, dtype=np.uint8)-1, :2].copy(), (-1, 2)) if player.sub_ids != None else None

            if player.superior_id == None and player.sub_ids == None:
                return None, None

            if player.superior_id == None:
                web = sub_pos
                alive = self.return_alive(np.asarray(player.sub_ids, dtype=np.uint8))
                web = web[alive==1]
            elif player.sub_ids == None:
                web = superior_pos
                alive = self.return_alive(player.superior_id)
                if not alive:
                    web = None
            else:
                try:
                    web = np.concatenate([superior_pos, sub_pos], axis=0)
                    alive = np.concatenate([[self.return_alive(player.superior_id)], self.return_alive(player.sub_ids)], axis=0)
                    web = web[alive==1]
                except Exception as e:
                    if self.log:
                        print(f"{e}, superior shape: {superior_pos.shape}, sub_pos shape: {sub_pos.shape}, superior id alive: {self.return_alive(player.superior_id)}, sub id alive: {self.return_alive(np.asarray(player.sub_ids, dtype=np.uint8))}")

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
    def get_ks_ms(self, player):
        web, mag = self.get_web_and_mag(player)
        if web is None:
            return None, None
        player_velocity = self.board_sight[player.id-1, 2:4].copy()
        player_k = self.k_ids[player.id-1].copy()
        sub_k = self.k_ids[np.asarray(player.sub_ids, dtype=np.uint8)-1].copy() if player.sub_ids != None else None
        superior_mass = self.masses[player.superior_id-1].copy() if player.superior_id != None else None
        sub_masses = self.masses[np.asarray(player.sub_ids, dtype=np.uint8)-1].copy() if player.sub_ids != None else None
        ks = None
        ms = None
        if player.superior_id == None:
            ks = sub_k
            ms = sub_masses
            alive = self.return_alive(np.asarray(player.sub_ids, dtype=np.uint8))
            ks = ks[alive==1]
            ms = ms[alive==1]
        elif player.sub_ids == None:
            ks = np.array([player_k], dtype=np.float16)
            ms = np.array([superior_mass], dtype=np.float16)
        else:
            ks = np.concatenate([np.asarray([player_k], dtype=np.float16), sub_k], axis=0)
            ms = np.concatenate([np.asarray([superior_mass], dtype=np.float16), sub_masses], axis=0)
            alive = np.concatenate([np.asarray([self.return_alive(player.superior_id)], dtype=np.float16), self.return_alive(np.asarray(player.sub_ids, dtype=np.float16))], axis=0)
            ks = ks[alive==1]
            ms = ms[alive==1]
        return ks, ms
class Setup_Swarm_Intelligence(Setup_Web):
    def __init__(self, **kwargs):
        Setup_Web.__init__(self, **kwargs)
    def get_spring(self, player, epsilon=10**-50):
        web, mag = self.get_web_and_mag(player)
        if web is None:
            return np.asarray([0,0], dtype=np.float16)
        player_velocity = self.board_sight[player.id-1, 2:4].copy()
        player_k = self.k_ids[player.id-1].copy()
        sub_ids = np.array(player.sub_ids, dtype=np.uint8)  if player.sub_ids != None else None
        sub_k = self.k_ids[sub_ids-1].copy() if player.sub_ids != None else None
        superior_mass = self.masses[player.superior_id-1].copy() if player.superior_id != None else None
        sub_masses = self.masses[sub_ids-1].copy() if player.sub_ids != None else None
        ks = None
        ms = None
        if player.superior_id == None:
            ks = sub_k
            ms = sub_masses
            alive = self.return_alive(sub_ids)
            ks = ks[alive==1]
            ms = ms[alive==1]
        elif player.sub_ids == None:
            ks = player_k
            ms = superior_mass
        else:
            ks = np.concatenate([np.asarray([player_k], dtype=np.float16), sub_k], axis=0)
            ms = np.concatenate([np.asarray([superior_mass], dtype=np.float16), sub_masses], axis=0)
            alive = np.concatenate([np.asarray([self.return_alive(player.superior_id)], dtype=np.uint8), self.return_alive(player.sub_ids)], axis=0)
            ks = ks[alive==1]
            ms = ms[alive==1]

        player_mass = self.masses[player.id-1].copy()
        mag[mag == 0] = 1
        try:
            radiis = np.diag(player.r_a/(mag)) @ web
        except Exception as e:
            if self.log:
                print(f"{e}, mag: {mag}, web: {web}")
        ks = np.reshape(ks, [-1])
        ms = np.reshape(ms, [-1])
        try:
            force = np.diag(ks) @(web-radiis*2)- 2*np.reshape(np.sqrt(ks*(ms+player_mass)/(ms*player_mass)), (-1, 1))@np.reshape(player_velocity, (1,2))
        except Exception as e:
            if self.log:
                print(f"{e}, web: {web}, web shape: {web.shape}, radiis: {radiis}, radiis shape: {radiis.shape}, ms: {ms}, ms shape: {ms.shape}, ks: {ks}, ks shape: {ks.shape}")
        force = np.sum(force, axis=0)
        above_limit = np.abs(force[np.abs(force) > self.spring_force_prop*player.force_prop])
        if above_limit.shape[0] is not 0:
            force /= above_limit.max()
            force *= self.spring_force_prop
        return force
    def get_spring_matrix(self):
        raise NotImplementedError
        living = self.get_alive_mask() == 1
        self.k_mat[:] = 0
        self.m_mat[:] = 0
        self.web_mat[:] = 0
        self.mag_mat[:] = 0
        k = -1
        for i in range(self.sides):
            for j in range(self.players_per_side[i]):
                if not self.player_array[i][j].alive:
                    continue
                k += 1
                player = self.player_array[i][j]
                id = player.id
                ks, ms = self.get_ks_ms(player)
                if ks is None or ms is None:	
                    continue
                self.k_mat[k, :ks.shape[0]] = ks
                self.m_mat[k, :ms.shape[0]] = ms
                
                if self.webs[id-1] is None or self.mags[id-1] is None:
                    continue
                self.web_mat[k, :self.webs[id-1].shape[0]] = self.webs[id-1]
                self.mag_mat[k, :self.mags[id-1].shape[0]] = self.mags[id-1]
    def get_springs(self):
        raise NotImplementedError
        self.get_spring_matrix()
        living = self.get_alive_mask() == 1
        player_mass_mat = self.masses[living]
        mag_mat = self.mag_mat[living].copy()
        radiis = self.r_as[living].copy()
        mag_mat_mask = mag_mat == 0
        mag_mat[mag_mat_mask] = 1 
        radii_denom = 1/mag_mat
        radii_denom[mag_mat_mask] = 0
        radiis = np.einsum("i,i...->i...", radiis, radii_denom)#radiis has shape [num_living, 2*num_subs-1]
        radiis = 2*np.einsum("ij, ij...->ij...", radiis, self.web_mat[living])#web_mat has shape [num_living, 2*num_subs-1, 2]
        #k_mat has shape [num_living, 2*num_subs-1] as well as m_mat

        forces = np.einsum("ij,ij...->ij...", self.k_mat[living], self.web_mat[living]-radiis)
        mass_denom = np.einsum("i...,i->i...", self.m_mat[living], self.masses[living])

        #if mass denom evaluates to 0, as the mass of the superior/subordinates are set to be None, the k_mat which this is later
        #multiplied by will evaluate to 0 too. Thus, set to 1 to avoid complications
        mass_num = np.transpose(np.transpose(self.m_mat[living])+self.masses[living])
        mass_denom_mask = mass_denom == 0
        mass_denom[mass_denom_mask] = 1
        mass_denom = 1/mass_denom
        mass_denom[mass_denom_mask] = 0
        mass_factor = np.einsum("ij,ij->ij", mass_num, mass_denom)
        damping = -2*np.sqrt(np.einsum("ij,ij->ij", self.k_mat[living], mass_factor))
        damping = np.einsum("ij,ik->ijk", damping, self.board_sight[living, 2:4])
        forces += damping
        forces = np.sum(forces, axis=1)
        k = 0
        for i in range(self.sides):
            for j in range(self.players_per_side[i]):
                player = self.player_array[i][j]
                if not player.alive:
                    continue
                force = forces[k]
                above_limit = np.abs(force[np.abs(force) > self.spring_force_prop*player.force_prop])
                if above_limit.shape[0] is not 0:
                    force /= above_limit.max()
                    force *= self.spring_force_prop
                forces[k] = force
                k += 1
        return forces
class Setup_Drag_Force(Setup_Swarm_Intelligence):
    def __init__(self, **kwargs):
        Setup_Springs.__init__(self, **kwargs)
    def get_drag(self, player, force):
        if self.get_height(*player.position) != 0:
            return np.asarray([0,0], dtype=np.float16)
        above_limit = np.abs(force[np.abs(force) > self.drag_force_prop])
        if above_limit.shape[0] is not 0:
            force /= above_limit.max()
            force *= self.drag_force_prop
        return force

class Setup_Rotate_Force(Setup_Drag_Force):
    def __init__(self, **kwargs):
        Setup_Drag_Force.__init__(self, **kwargs)
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
        force_3d = force_3d_unit*(force_mag+cos_weight)
        return force_3d[:2]
