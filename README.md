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
python -m tactic_game_gym.tactic_game.game --test_env --show --log --use_boid --full_view
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
