# tactic_game_gym

A gym environment for tactic_game

# Installation instructions

Just run

```
pip install -e .
```

# Test environment

Run

```
python -m tactic_game_gym.tactic_game.game --test_env --show --log --full_view --num_types=1 --infantry_prop=1 --archer_prop=0 --cavarly_prop=0 --wall_prop=0
```

# Get environment

In python code, write

```
imporg gym
import tactic_game_gym
env = gym.make("tactic_game-v0")
```

and you should be able to get your environment!

# The great class chain

args_env -> map_env -> setup_env -> game_base_env -> swarm_intelligence_env -> physics_env -> game_utility -> gym_env

# Baseline play command
 python -m baselines.run --alg=ppo2 --env=tactic_game-v0 --network=conv_only --log_path=logs_ppo2_conv_only_faster_random_act_1_rad_2_infantry_v5_multi_category --save_video_interval=10 --save_path=save --reward_scale=0.1 --ent_coef=0.01 --num_env=8 --save_video_length=1000 --save_interval=10 --log_interval=1 --act_board_size=1 --game_timestep=0.2 --is_train --strength=0.35 --full_view --max_players=100 --moves_without_model=10 --obs_board_size=64 --board_size=64 --play --show
