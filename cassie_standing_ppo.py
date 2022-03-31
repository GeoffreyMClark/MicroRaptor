from math import gamma
import cv2
import gym
import time
import torch as th

import tensorboard

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from cassie.envs.cassie import CassieEnv

from cassie_render_callback import RenderCallback


if __name__ == '__main__':
    eval_env = CassieEnv()
    env_id = 'CassieEnv-v0'
    num_cpu = 8  # Number of processes to use
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    env = make_vec_env(env_id, n_envs=num_cpu, gamma=0.997, seed=0, vec_env_cls=DummyVecEnv)
    # Custom actor (pi) and value function (vf) networks
    # of two layers of size 32 each with Relu activation function
    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[512, 256], vf=[512, 256])])
    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs/")

    render_call = RenderCallback(render_freq=10000, env=eval_env)

    for i in range(int(1e6)):
        model.learn(total_timesteps=int(2e10), callback=render_call)
        model.save("cassie_standing{i}")


        # for _ in range(3):
        #     obs = eval_env.reset()
        #     done= False
        #     while done == False:
        #         action, _states = model.predict(obs)
        #         obs, reward, done, info = eval_env.step(action)
        #         raw=eval_env.render(mode='rgb_array')
        #         cv2.imshow("cassie_standing_model",raw)
        #         cv2.waitKey(1)