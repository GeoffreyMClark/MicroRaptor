import gym
import numpy as np
import time
import cv2
from cheetah.envs.cheetah import CheetahEnv
from tf_agents.environments import suite_gym

# This should always work
env = CheetahEnv()
obs = env.reset()


Kp = 1
Kd = .5
for i in range(1000000000):
    env.render()
    # camera = env.render(mode="rgb_array", camera_id=0)
    # cv2.imshow("camera 0", camera)
    # cv2.waitKey(1)

    # t1 = np.sin(i/100)*10
    # t2 = np.sin(i/115)*10
    # t3 = np.sin(i/124)*10
    # t4 = np.sin(i/133)*10
    # t5 = np.sin(i/142)*10

    action = np.asarray([0,0,0,0,0,0,0,0,0,0,0,0])

    obs, reward, done, info = env.step(action)
    if i==10:
        pass
env.close()




