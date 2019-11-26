from gym.envs.registration import register

register(
    id='tactic_game_v0',
    entry_point='game_env:Game_Env_v0',
)