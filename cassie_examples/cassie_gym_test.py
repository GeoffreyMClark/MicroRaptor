import gym
import numpy as np
import time
from cassie import CassieEnv
from tf_agents.environments import suite_gym

# This should always work
env = CassieEnv()
obs = env.reset()
prev_pose = np.asarray([3, 0, 10, -25, -85,   -3, 0, 10, -25, -85])

Kp = 1
Kd = .5
for i in range(1000000000):
    env.render()

    print(obs[3:6])

    goal_pose = np.asarray([3, 0, 10, -25, -85,  -3, 0, 10, -25, -85])
    actual_pose = np.asarray([obs[0],obs[1],obs[2],obs[3],obs[4],   obs[8],obs[9],obs[10],obs[11],obs[12]])
    
    P = (goal_pose-actual_pose)*Kp
    D = (prev_pose-actual_pose)*Kd
    action = P 

    prev_pose = actual_pose
    obs, reward, done, info = env.step(action)
env.close()


# Test tf-agents
# env = CassieEnv()
# env.reset()
# action = env.action_space.sample()
# print('Observation Spec:')
# print(env.time_step_spec().observation)
pass




