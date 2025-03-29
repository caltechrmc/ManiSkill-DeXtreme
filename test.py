from tasks.reorient_cube import *

import gymnasium as gym
import numpy as np

env = gym.make("ReorientCube-v0", render_mode="human")
env.reset()

while True:
    env.step(None)
    env.render()