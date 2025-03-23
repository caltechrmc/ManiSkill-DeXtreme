from tasks.cube_reorientation import *
import matplotlib.pyplot as plt
import gymnasium as gym

env = gym.make("ReorientCube-v1", render_mode="human")
env.reset()

while True:
    action = env.action_space.sample()
    env.step(action)
    env.render()
#img = env.render()
#plt.imshow(img[0])
#plt.show()