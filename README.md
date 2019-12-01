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
python -m tactic_game_gym.tactic_game.game --test_env --show
```
# Get environment
In python code, write
```
imporg gym
import tactic_game_gym
env = gym.make("tactic_game-v0")
```
and you should be able to get your environment!
