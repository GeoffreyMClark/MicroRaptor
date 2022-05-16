import gym
import numpy as np
import time
import cv2
from cassie.envs.cassie import CassieEnv
from tf_agents.environments import suite_gym

# This should always work
env = CassieEnv()
obs = env.reset()
prev_pose = np.asarray([3, 0, 10, -25, -85,   -3, 0, 10, -25, -85])

Kp = 1
Kd = .5
for i in range(1000000000):
    camera = env.render(mode="rgb_array", camera_id=0)
    cv2.imshow("camera 0", camera)
    cv2.waitKey(1)

    # print(obs[3:6])

    goal_pose = np.asarray([3, 0, 10, -25, -85,  -3, 0, 10, -25, -85])
    actual_pose = np.asarray([obs[0],obs[1],obs[2],obs[3],obs[4],   obs[8],obs[9],obs[10],obs[11],obs[12]])
    
    P = (goal_pose-actual_pose)*Kp
    D = (prev_pose-actual_pose)*Kd
    action = P 

    t1 = np.sin(i/100)*10
    t2 = np.sin(i/115)*10
    t3 = np.sin(i/124)*10
    t4 = np.sin(i/133)*10
    t5 = np.sin(i/142)*10

    action = np.asarray([10,10,10,10,10,10,10,10,10,10])


    prev_pose = actual_pose
    obs, reward, done, info = env.step(action)
    if done == True:
        pass
env.close()




# BALL PHYSICS
# ball mass = 0.1kg
# ball velocity = 1 m/s - 20 m/s
# contact time = 0.1 s
# force = m * (vel/time)
# 30 <= force <= 100 << 200



