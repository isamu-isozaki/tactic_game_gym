from tactic_game_gym.tactic_game.env_classes.game_utility import Playable_Game
from tactic_game_gym.tactic_game.utility import get_n_colors
import numpy as np

class Gym_Env(Playable_Game):
    def __init__(self, **kwargs):
        Playable_Game.__init__(self, **kwargs)
    def update_step(self):
        self.game_step()
        self.reset_web()
        self.set_board()
        done = self.end()
        if self.t>=self.terminate_turn or self.end():
            done = True
        
        self.t += 1
        self.finished_sides[...] = 0
        self.get_sight()
        dones = [done for _ in range(self.sides)]
        infos = [{} for _ in range(self.sides)]
        if self.is_train:
            if (self.stage != self.act_board_size) and ((self.total_moves+1) % self.stage_update_num == 0):
                if self.log:
                    print(f"stage is {self.stage}")
                self.stage *= 2
            self.total_moves += 1
        return self.obs, self.rewards, dones, infos
    def step(self, action):
        if not self.started:
            self.start_game()
        side = self.side
        action = np.reshape(action, [self.act_board_size, self.act_board_size, self.num_types, 2])
        action *= self.player_force_prop
        if self.is_train:
            size = self.act_board_size // self.stage
            for i in range(self.stage):
                for j in range(self.stage):
                    for l in range(self.num_types):
                        for k in range(2):
                            action_segment = action[size*i:size*(i+1), size*j:size*(j+1), l,  k]
                            action_mean = action_segment.mean()
                            action_std = action_segment.std()
                            action[size*i:size*(i+1), size*j:size*(j+1), l, k] = np.random.normal(action_mean, action_std)
        self.action[self.side] = action.copy()
        for i in range(self.num_types):
            self.move_board[side, i] = cv2.resize(action[:, :, i, :].astype(np.float32), (self.board_size[0], self.board_size[1])).astype(np.float16)
        # if self.save_imgs:
        # 	self.show_board(folder=self.base_directory   + f"/animation/animation_players_{len(folders)//2}",save=self.save_animation, step=t, title="Moves of {}".format(self.remaining_players))
        # 	self.show_interact_board(folder=self.base_directory   + f"/animation/animation_interact_{len(folders)//2}",save=self.save_animation, step=t, title="Moves of {}".format(self.remaining_players))
        self.finished_sides[side] = 1
        self.side += 1
        self.side %= self.sides
        if self.finished_sides[self.finished_sides == 0].shape[0] == 0:
            return self.update_step()
        return None, None, None, None
    def reset(self):
        self.__init__(**self.kwargs)
        self.start_game()
        self.vel_mags = np.zeros(self.player_num, dtype=np.float16)
        self.t = 0
        self.get_sight()
        return [self.obs]
    def render(self, mode='human', close=False):
        self.render_output = self.beautiful_output.copy()
        self.arrow_output = np.zeros_like(self.render_output, dtype=np.float16)
        #self.screen.blit(self.surf, (0,0))
        #draw_width = self.draw_width
        colors = get_n_colors(self.sides)
        for i in range(self.sides):
            for i2 in range(self.sides):
                for j in range(self.players_per_side[i2]):
                    player = self.player_array[i2][j]
                    if not player.alive or not (self.full_view or self.can_see[i, player.id]):
                        continue
                    try:
                        xys = self.board_sight[player.id, :4].copy()
                        x = xys[:2]
                        y = xys[2:]
                        y += x
                        x = self.switch_to_pymunk(x)
                        y = self.switch_to_pymunk(y)
                        cv2.circle(self.render_output[i], tuple(self.switch_to_pymunk([int(player.position[0]), int(player.position[1])])), int(player.radius) if int(player.radius) > 0 else 1, tuple([int(m) for m in colors[i2]]))
                        cv2.line(self.render_output[i], tuple([int(x[0]), int(x[1])]), tuple([int(y[0]), int(y[1])]), tuple([int(m) for m in colors[i2]]))
                    except Exception as e:
                        print(f"{e}. color: {color}. position: {player.position}, radius: {player.radius}, alive: {player.alive}")
                        import traceback

                        print(traceback.format_exc())
            arrow_size = self.board_size[0]/(self.stage*2)
            size = self.act_board_size//self.stage
            for a0 in range(0,self.stage):
                for a1 in range(0,self.stage):
                    arrow = self.action[i, a0*size, a1*size].copy()
                    arrow *= arrow_size
                    x=a0*2+1
                    y=a1*2+1
                    start = self.switch_to_pymunk([int(x*arrow_size), int(y*arrow_size)])
                    end = self.switch_to_pymunk([int(x*arrow_size+arrow[0]), int(y*arrow_size+arrow[1])])

                    cv2.arrowedLine(self.arrow_output[i], tuple(start), tuple(end), (255, 255, 255))
        self.render_output = np.concatenate([self.render_output, self.arrow_output], axis=1)
        return [self.render_output]

    def end(self):
        #if np.max(self.remaining_players) == np.sum(self.remaining_players):
        #   return True#only one side remains
        if np.sum(self.remaining_players) < np.sum(self.players_per_side)*self.min_frac or np.max(self.remaining_players) == np.sum(self.remaining_players):#If the total number is half,
            return True
        for i in range(self.sides):
            if self.remaining_players[i] < self.min_players:
                return True
        return False