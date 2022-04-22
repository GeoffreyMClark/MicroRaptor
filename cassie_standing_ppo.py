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
    env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=DummyVecEnv)
    policy_kwargs = dict(
                    log_std_init=-2,
                    ortho_init=False,
                    activation_fn=th.nn.ReLU, 
                    net_arch=[dict(pi=[1024, 512], vf=[1024, 512])])

    # TEST batch_size=512-5120    n_epochs=10
    model = PPO('MlpPolicy', env, learning_rate=0.0001, n_steps=4096, batch_size=512, n_epochs=10, clip_range=0.2, gamma=0.998, gae_lambda=0.99, 
    use_sde=False, ent_coef=0.0, create_eval_env=False, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs/")

    render_call = RenderCallback(render_freq=10000, env=eval_env)

    model.learn(total_timesteps=int(2e15), callback=render_call)
    # model.learn(total_timesteps=int(2e10))
    model.save("cassie_standing{i}")
