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
                    net_arch=[dict(pi=[128,128],vf=[128, 128])])

    # PPO_10 - no perturbations | pi=[1024, 512],vf=[1024, 512] | failed_reward=-100.0
    # model = PPO('MlpPolicy', env, learning_rate=0.0001, n_steps=2048, batch_size=128, n_epochs=10, clip_range=0.2, gamma=0.998, gae_lambda=0.99, 
    # use_sde=False, create_eval_env=False, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs/")

    # PPO_11 - no perturbations | pi=[1024, 512],vf=[1024, 512] | failed_reward=-100.0
    # model = PPO('MlpPolicy', env, learning_rate=0.0001, n_steps=2048, batch_size=128, n_epochs=10, clip_range=0.2, gamma=0.998, gae_lambda=0.95, 
    # use_sde=False, create_eval_env=False, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs/")

    # PPO_12 - no perturbations | pi=[1024, 512],vf=[1024, 512] | failed_reward=-100.0
    # model = PPO('MlpPolicy', env, learning_rate=0.0001, n_steps=2048, batch_size=128, n_epochs=10, clip_range=0.2, gamma=0.998, gae_lambda=0.99, 
    # use_sde=False, create_eval_env=False, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs/")

    # PPO_13 - 3-20 (2s) perturbations | pi=[1024, 512],vf=[1024, 512] | failed_reward=-100.0
    # model = PPO('MlpPolicy', env, learning_rate=0.0001, n_steps=2048, batch_size=128, n_epochs=10, clip_range=0.2, gamma=0.998, gae_lambda=0.98, 
    # use_sde=False, create_eval_env=False, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs/")

    # PPO_14 - no perturbations | pi=[512, 256],vf=[512, 256] | failed_reward=-500.0
    # model = PPO('MlpPolicy', env, learning_rate=0.0001, n_steps=8192, batch_size=128, n_epochs=10, clip_range=0.2, gamma=0.999, gae_lambda=0.99, 
    # use_sde=False, create_eval_env=False, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs/")

    # PPO_15 - same as 14 but added _adjust_circle function to quats and orientations
    # model = PPO('MlpPolicy', env, learning_rate=0.0001, n_steps=8192, batch_size=128, n_epochs=10, clip_range=0.2, gamma=0.998, gae_lambda=0.99, 
    # use_sde=False, create_eval_env=False, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs/")

    # PPO_16 - no perturbations | pi=[128, 128],vf=[128, 128])] | failed_reward=-500.0 | max reward changed to 1 with no survival reward | activation_fn=th.nn.thanh
    # model = PPO('MlpPolicy', env, learning_rate=0.0001, n_steps=8192*2, batch_size=256, n_epochs=10, clip_range=0.25, gamma=0.998, gae_lambda=0.99, 
    # use_sde=False, create_eval_env=False, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs/")

    # PPO_17 - no perturbations | pi=[128, 128],vf=[128, 128])] | failed_reward=-500.0 | max reward changed to 1 with no survival reward | activation_fn=th.nn.thanh
    # model = PPO('MlpPolicy', env, learning_rate=0.0002, n_steps=512, batch_size=256, n_epochs=10, clip_range=0.25, gamma=0.998, gae_lambda=0.9, 
    # use_sde=False, create_eval_env=False, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs/")

    # PPO_new_4 - no perturbations | pi=[128,128],vf=[128, 128]] | failed_reward=-500.0/-1
    # model = PPO('MlpPolicy', env, learning_rate=0.0001, n_steps=512, batch_size=128, n_epochs=10, clip_range=0.25, gamma=0.999, gae_lambda=0.99, 
    # use_sde=False, create_eval_env=False, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs/")

    # PPO_new_5 - no perturbations | pi=[128,128],vf=[128, 128]] | failed_reward=-500.0/-1
    # dont remember exactly if relu or tanh was used and am 90% sure gae_lambda=0.90
    model = PPO('MlpPolicy', env, learning_rate=0.0001, n_steps=8192, batch_size=128, n_epochs=10, clip_range=0.2, gamma=0.999, gae_lambda=0.90, 
    use_sde=False, create_eval_env=False, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs/")

    render_call = RenderCallback(render_freq=10000, env=eval_env)

    for i in range(int(1e6)):
        model.learn(total_timesteps=int(2e10), callback=render_call, eval_env=eval_env, eval_freq=10000)   #, callback=render_call
        model.save("cassie_standing{i}")
