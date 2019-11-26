from gym.envs.registration import register

register(
    id='tactic_game-v0',
    entry_point='tactic_game_gym.envs:Game_Env',
)