from tactic_game_gym.tactic_game.env_classes.physics_env import Setup_Rotate_Force
import numpy as np
from tqdm import tqdm
import itertools

import pygame
from pygame.locals import *
class Move(Setup_Rotate_Force):
    def __init__(self, **kwargs):
        Setup_Rotate_Force.__init__(self, **kwargs)
    def move(self):
        living = self.get_alive_mask() == 1
        final_vel = self.board_sight[living, 13:15].copy()
        final_vel *= self.max_speed#For now only one max speed
        #forces = np.einsum("i..., i->i...", forces, self.player_forces[living])

        #drag_forces = -np.einsum("i,i...->i...", self.vel_mags[is_drag], self.board_sight[is_drag, 2:4])
        #total_mag = (self.player_force*self.player_force_prop+self.align_force_prop+self.cohesion_force_prop)
        #At half of max velocity, damping in velocity d
        #forces -= self.board_sight[living, 2:4]*total_mag/(self.max_speed*self.damping_start)
        if self.use_boid:
            forces += self.get_boids(living)
        forces = (final_vel - self.board_sight[living, 2:4])*self.vel_diff_coef

        z = np.abs(self.board_sight[living, 11].copy())#z index of rotated 3d array
        sign = np.sign(np.einsum("...i, ...i->...", forces, self.board_sight[living, 15:17]))
        z = np.einsum("i,i->i", z, sign)
        k = 0
        for i in range(self.sides):
            for j in range(self.players_per_side[i]):
                if not self.player_array[i][j].alive:
                    continue
                player = self.player_array[i][j]
                force = forces[k]
                if self.use_spring:
                    force += self.get_spring(player)
                #Return later
                #force = self.rotate_force(player, force, z[k])

                try:
                    self.player_array[i][j].apply_force(force, self.current_balls[k])
                except Exception as e:
                    if self.log:
                        print(f"Exception: {e}, i: {i}, j: {j}, k: {k}, id: {player.id}, current_balls length: {len(self.current_balls)}")
                k += 1
        k = 0
        for i in range(self.sides):
            for j in range(self.players_per_side[i]):
                id = self.player_array[i][j].id
                if not self.player_array[i][j].alive:
                    self.vel_mags[id] = 0
                    continue
                position = self.current_balls[k].body._get_position()
                speed = self.current_balls[k].body.velocity.length
                velocity = np.asarray(self.current_balls[k].body._get_velocity(), dtype=np.float32)
                if speed > self.max_speed:
                    if speed == np.inf:
                        print(f"speed for player {i} {j} is infinity")
                    velocity /= speed
                    velocity *= self.max_speed
                    self.current_balls[k].body._set_velocity(velocity.tolist())
                    speed = self.max_speed
                self.player_array[i][j].speed = speed
                self.vel_mags[id] = speed
                self.player_array[i][j].vel = [self.player_array[i][j].position, self.player_array[i][j].position+velocity]
                self.player_array[i][j].position = np.array(position, dtype=np.float32)
                self.player_array[i][j].velocity = velocity
                k += 1


        self.set_board()#update the board
class Mobilize(Move):
    def __init__(self, **kwargs):
        Move.__init__(self, **kwargs)
    def mobilize_step(self):
        self.move()
        self.render_game()
    def mobilize(self):
        self.start_game()
        if self.log:
            print(f"Game setup completed at {time.time()-self.start}")
        # if self.save_imgs:
        # 	if not os.path.exists( self.base_directory   + "/animation"):
        # 			os.makedirs( self.base_directory   + "/animation")
        # 	folders = os.listdir(self.base_directory + "/animation")
        self.vel_mags = np.zeros(self.player_num, dtype=np.float32)
        for t in tqdm(itertools.count()):
            try:
                for event in pygame.event.get():
                    if event.type == MOUSEBUTTONUP:
                        None
                    elif event.type == pygame.QUIT:
                        break
                self.mobilize_step()
                # if self.save_imgs:
                # 	self.show_board(folder=self.base_directory   + f"/animation/animation_{len(folders)}",save=self.save_animation, step=t, title="Moves of {}".format(self.remaining_players))
                self.reset_web()
                if t>=self.terminate_turn:
                    break
            except KeyboardInterrupt:
                break
class Attack(Mobilize):
    def __init__(self, **kwargs):
        Mobilize.__init__(self, **kwargs)
    def remove_glitched_players(self):
        for i in range(self.sides):
            for j in range(self.players_per_side[i]):
                player = self.player_array[i][j]
                if player.alive:
                    pos = player.position
                    if (0<= int(pos[0]) <= self.board_size[0]) and (0<= int(pos[1]) <= self.board_size[1]):
                        continue
                    else:
                        print(f"Glitched player at {pos}")
                        self.player_array[i][j].damage(player.hp)
                        self.player_array[i][j].alive = False
                        self.remaining_players[i] -= 1
                        continue
    def attack(self, epsilon=1e-5):
        # Change attack to constant size
        INF = 10**5
        rotation = True
        prop_side = self.prop_side
        positions = self.board_sight[:, :2].copy()
        velocities = self.board_sight[:, 2:4].copy()
        living = self.get_alive_mask() == 1
        is_wall = self.player_type_mask == 3#Walls can't attack
        is_archer = self.player_type_mask == 0#Archer's range is higher when staying still
        alive = []
        for i in range(self.sides):
            for j in range(self.players_per_side[i]):
                if self.player_array[i][j].alive:
                    alive.append(self.player_array[i][j].id)
        mags = self.vel_mags.copy()
        mags = mags[living]
        positions = positions[living]
        velocities = velocities[living]
        num_players = np.sum(self.remaining_players)
        attacked = np.zeros(num_players, dtype=np.float32)
        mags_temp = mags.copy()
        mags_temp[mags_temp == 0] = 1
        cos_sin = (velocities) / (mags_temp[:, None])
        rot_matrix = np.concatenate([\
                            np.concatenate([cos_sin[:, 0, None, None],\
                                             cos_sin[:, 1, None, None]], axis = 2),\
                            np.concatenate([cos_sin[:, 1, None, None],\
                                            -cos_sin[:, 0, None, None]], axis = 2)], axis = 1)
        x, y = self.board_sight[living, 0], self.board_sight[living, 1]
        X = np.reshape(-positions[:, 0, None]+x, [num_players, num_players])
        #The x coordinate of all other players from each player
        Y = np.reshape(-positions[:, 1, None]+y, [num_players, num_players])
        #X[i], Y[i] is the x and y coordinates of all of the players when the ith player is at (0,0)
        XY = np.concatenate([X[..., None], Y[..., None]], axis = 2)
        attack_range_mags = np.ones_like(mags)
        #ignore archers for now
        #attack_range_mags[is_archer[living]] *= 
        base_index = 12
        attack_range = self.range_factor*np.einsum("i,i->i",attack_range_mags, self.board_sight[living, base_index])[:, None]#set max value of einsum to 1
        #attack_range = np.ones(num_players) + 10
        #if self.log:
        # print(f"Mean attack range: {attack_range.mean()}")
        max_attack_range = np.max(attack_range)
        if rotation:
            #rot_matrix has shape [num_players, 2, 2]
            #XY has shape [num_players, num_players, 2]
            XY_rot = np.einsum("mij,m...j->i...m", rot_matrix, XY)
            #XY_rot has shape [2, num_players, num_players]
            #This step leads to duplicates forming at different locations
            Z = XY_rot.copy()#for debuggin
            Z[0][Z[0] < 0] = INF#If out of range, make sure can't be reached
            XY_rot_abs = np.abs(Z)
            Y_rot_abs_passive = np.abs(XY_rot[1])#quite sure there's a faster way but go with this for now
        else:
            XY_rot_abs = np.abs(XY)

        if rotation:
            X_end = XY_rot_abs[0]
            Y_end = XY_rot_abs[1]
            Y_end_passive = Y_rot_abs_passive
        else:
            X_end = np.reshape(XY_rot_abs[..., 0], [num_players, num_players])
            Y_end = np.reshape(XY_rot_abs[..., 1], [num_players, num_players])
            Y_end_passive = Y_end

        XY_out = X_end + (1/prop_side)*Y_end
        XY_out_passive = X_end+Y_end_passive
        masks = np.transpose(np.transpose(XY_out, [1,0]) <= attack_range, [1,0]) | (XY_out_passive <= self.passive_range)
        masks = np.transpose(masks, [1,0])
        self.attacked[...] = 0
        self.attacked_dist[...] = 0
        self.can_see[...] = 0
        if self.init_reward:
            self.hard_coded_rewards['death_offset'][...] = 0
            self.hard_coded_rewards['damage'][...] = 0
            self.hard_coded_rewards['seen'][...] = 0
            self.dead[...] = 0
            self.rewards[...] = 0
            self.init_reward = False
        k = -1
        for i in range(self.sides):
            for j in range(self.players_per_side[i]):
                player = self.player_array[i][j]
                if not player.alive:
                    continue
                k += 1
                """self.attacked[i] denotes the amount of damage the side i inflicts on the other sides"""
                
                living_k = living.copy()
                living_k[living_k] = masks[k]
                if self.vel_mags[player.id] < 0.01:
                    continue
                if player.type == 1 and self.attack_turn % self.archer_freq != 0:
                    self.attacked[i, living_k & (self.player_sides != i)] += epsilon
                elif player.type == 3:
                    self.attacked[i, living_k & (self.player_sides != i)] += epsilon
                else:
                    self.attacked[i, living_k & (self.player_sides != i)] += player.strength
        #Potenially slow bad but
        #if self.log:
        # print(self.attacked)
        self.attacked_dist = self.attacked.copy()
        #self.attacked contains damaged done from side i on all players
        self.can_see = self.attacked > 0
        for i in range(self.sides):
            self.can_see[i] = self.can_see[i] | (self.player_sides == i)
        
                
        #can_see contains all players each side can see
        attacked_sum = np.sum(self.attacked, axis = 0)
        #
        for i in range(self.sides):
            # Damage done to side i players from all the other players
            self.attacked[i] = attacked_sum - self.attacked[i]
        k = 0
        temp_rewards = dict(self.hard_coded_rewards)
        temp_dead = np.zeros(self.sides, dtype=np.float32)
        temp_rewards['damage'] = np.zeros(self.sides, dtype=np.float32)
        temp_rewards['seen'] = np.zeros(self.sides, dtype=np.float32)
        for i in range(self.sides):
            for j in range(self.players_per_side[i]):
                if not self.player_array[i][j].alive:
                    continue
                player_id = self.player_array[i][j].id
                player_hp = self.player_array[i][j].hp
                player_alive = self.player_array[i][j].damage(self.attacked[i][player_id]+self.continue_penalty)
                temp_rewards['damage'][i] += np.min([self.attacked[i][player_id], player_hp])
                temp_rewards['seen'][i] += 1 if self.attacked[i][player_id] > 0 else 0
                if not player_alive:
                    temp_dead[i] += 1
                    self.remaining_players[i] -= 1
                k += 1
        total_death = np.sum(temp_dead)
        total_damage = np.sum(temp_rewards['damage'])
        total_seen = np.sum(temp_rewards['seen'])

        for i in range(self.sides):
            self.hard_coded_rewards['death_offset'][i] += total_death - 2*temp_dead[i]#how many more died then your side
            self.hard_coded_rewards['damage'][i] += total_damage - 2*temp_rewards['damage'][i]#how much more damage occured on your side
            self.hard_coded_rewards['seen'][i] += total_seen - 2*temp_rewards['seen'][i]#how many more were seen on your side
            #Overall, becomes a zero sum game
            #self.rewards[i] *= 1 if self.rewards[i] > 0 else self.penalty_discount
        self.remove_glitched_players()
        for i in range(self.sides):
            for j in range(self.players_per_side[i]):
                if self.player_array[i][j].alive and self.player_array[i][j].id in alive:
                    alive.remove(self.player_array[i][j].id)
        for id in alive:#all dead
            self.space.remove(self.balls[id], self.balls[id].body)
            self.current_balls.remove(self.balls[id])



        self.set_board()
        self.attack_turn += 1
        #change reward
class Playable_Game(Attack):
    def __init__(self, **kwargs):
        Attack.__init__(self, **kwargs)
    def game_step(self):
        self.move()
        self.render_game()
        if self.t != 0 and self.t % self.moves_without_attack == 0:
            self.attack()
        self.reset_player_web()
    def reset_player_web(self):
        alive = self.get_alive_mask() == 1
        for i in range(self.sides):
            for j in range(self.players_per_side[i]):
                new_web = []
                for web_id in self.player_array[i][j].web:
                    if alive[web_id]:
                        new_web.append(web_id)
                self.player_array[i][j].web = new_web
    def env_step(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                break
            elif event.type == pygame.MOUSEBUTTONDOWN:#By clicking two times form arrow that gives force for players to move
                if event.button == 1:
                    if self.start_pos == None:
                        self.start_pos = pygame.mouse.get_pos()
                    else:
                        self.interact_move(self.start_pos, pygame.mouse.get_pos())
                        self.start_pos = None
                elif event.button == 4:
                    self.vec_width += self.vec_width_diff
                    if self.log:
                        print(f"Current vec width is {self.vec_width}")
                elif event.button == 5:
                    self.vec_width -= self.vec_width_diff
                    if self.log:
                        print(f"Current vec width is {self.vec_width}")
                elif event.button == 3:
                    self.clear_screen()
                    if self.log:
                        print(f"Cleared screen")
                elif event.button == 2:
                    self.interact_side += 1
                    self.interact_side %= self.sides
                    if self.log:
                        print(f"Changed side to {self.interact_side}")
            elif event.type == pygame.KEYDOWN:
                player_types = ["archer", "cavarly", "infantry", "wall"]
                if event.key == pygame.K_0:
                    if 0 not in self.type_to_index:
                        print("Can't change sides to this")
                    else:
                        self.interact_type = 0
                        if self.log:
                            print(f"Changed type to {player_types[self.interact_type]}")
                elif event.key == pygame.K_1:
                    if 1 not in self.type_to_index:
                        print("Can't change sides to this")
                    else:
                        self.interact_type = 1
                        if self.log:
                            print(f"Changed type to {player_types[self.interact_type]}")
                elif event.key == pygame.K_2:
                    if 2 not in self.type_to_index:
                        print("Can't change sides to this")
                    else:
                        self.interact_type = 2
                        if self.log:
                            print(f"Changed type to {player_types[self.interact_type]}")
                elif event.key == pygame.K_3:
                    if 3 not in self.type_to_index:
                        print("Can't change sides to this")
                    else:
                        self.interact_type = 3
                        if self.log:
                            print(f"Changed type to {player_types[self.interact_type]}")
            else:
                None
        self.t += 1
        
        
        
    def init_env(self):
        self.vel_mags = np.zeros(self.player_num, dtype=np.float32)
        self.start_pos = None
    def run_env(self):
        """
        Test if environment is running properly. Show argument must be true
        """
        self.init_env()
        self.start_game()
        while True:
            try:
                self.env_step()
                self.game_step()
                if self.t>=self.terminate_turn or self.end():
                    pygame.quit()
                
                # if self.save_imgs:
                # 	self.show_board(folder=self.base_directory   + f"/animation/animation_players_{len(folders)//2}",save=self.save_animation, step=t, title="Moves of {}".format(self.remaining_players))
                # 	self.show_interact_board(folder=self.base_directory   + f"/animation/animation_interact_{len(folders)//2}",save=self.save_animation, step=t, title="Moves of {}".format(self.remaining_players))

                #self.reset_web()
                self.set_board()
            except Exception as e:
                import traceback
                print(f"{e}")
                print(traceback.format_exc())
                pygame.quit()
                break
            if self.t>=self.terminate_turn or self.end():
                break