from tactic_game_gym.tactic_game.env_classes.web_env import Setup_Web
import numpy as np

class Setup_Springs(Setup_Web):
    def __init__(self, **kwargs):
        Setup_Web.__init__(self, **kwargs)
    def get_spring(self, player, epsilon=10**-50):
        web, mag = self.get_web_and_mag(player)
        if web is None:
            return np.asarray([0,0], dtype=np.float16)
        player_velocity = self.board_sight[player.id, 2:4].copy()
        player_k = self.k_ids[player.id].copy()
        sub_ids = player.sub_ids
        sub_k = self.k_ids[sub_ids].copy() if player.sub_ids != None else None
        superior_mass = self.masses[player.superior_id].copy() if player.superior_id != None else None
        sub_masses = self.masses[sub_ids].copy() if player.sub_ids != None else None
        ks = None
        if player.superior_id == None:
            ks = sub_k
            alive = self.return_alive(sub_ids)
            ks = ks[alive==1]
        elif player.sub_ids == None:
            ks = player_k
        else:
            ks = np.concatenate([np.asarray([player_k], dtype=np.float16), sub_k], axis=0)
            alive = np.concatenate([[self.return_alive(player.superior_id)], self.return_alive(player.sub_ids)], axis=0)
            ks = ks[alive==1]

        player_mass = self.masses[player.id].copy()
        mag[mag == 0] = 1
        try:
            radiis = np.diag(player.r_a/(mag)) @ web
        except Exception as e:
            if self.log:
                print(f"{e}, mag: {mag}, web: {web}")
        ks = np.reshape(ks, [-1])
        try:
            force = np.diag(ks) @(web-radiis*2)- self.spring_damping_factor*np.reshape(player_velocity, (1,2))
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
                
                if self.webs[id] is None or self.mags[id] is None:
                    continue
                self.web_mat[k, :self.webs[id].shape[0]] = self.webs[id]
                self.mag_mat[k, :self.mags[id].shape[0]] = self.mags[id]
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
    def get_ks_ms(self, player):
        web = self.get_web(player)
        if web is None:
            return None
        player_velocity = self.board_sight[player.id, 2:4].copy()
        player_k = self.k_ids[player.id].copy()
        sub_k = self.k_ids[player.sub_ids].copy() if player.sub_ids != None else None
        superior_mass = self.masses[player.superior_id].copy() if player.superior_id != None else None
        sub_masses = self.masses[player.sub_ids].copy() if player.sub_ids != None else None
        ks = None
        ms = None
        if player.superior_id == None:
            ks = sub_k
            ms = sub_masses
            alive = self.return_alive(player.sub_ids)
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