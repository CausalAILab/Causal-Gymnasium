import gymnasium as gym
import minigrid
import numpy as np
from gymnasium.utils.play import play
from causal_gym.envs import WindyMiniGrid

# left = 0
# right = 1
# forward = 2
# # Pick up an object
# pickup = 3
# # Drop an object
# drop = 4
# # Toggle/activate an object
# toggle = 5
# # Done completing task
# done = 6

lavagrid = gym.make('MiniGrid-LavaGapS6-v0', max_episode_steps=30, agent_pov=False, render_mode='rgb_array', highlight=False, tile_size=32)
windy_lavagrid = WindyMiniGrid(env=lavagrid,show_wind=True)
play(env=windy_lavagrid, 
     keys_to_action={
        "w": 2,
        "a": 0,
        "s": 6,
        "d": 1,
     },
     wait_on_player=True,
     seed=6458,
     zoom=4
     )