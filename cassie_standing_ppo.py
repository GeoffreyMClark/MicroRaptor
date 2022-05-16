from math import gamma
import cv2
import gym
import time
import torch as th
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorboard

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from cassie.envs.cassie import CassieEnv
from cassie_render_callback import RenderCallback


if __name__ == '__main__':
    eval_env = CassieEnv(max_episode_steps=100000)
    env_id = 'CassieEnv-v0'
    num_cpu = 64  # Number of processes to use
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=DummyVecEnv)
    env = make_vec_env(env_id, n_envs=num_cpu, vec_env_cls=DummyVecEnv)
    policy_kwargs = dict(
                    # log_std_init=-2,
                    ortho_init=False,
                    activation_fn=th.nn.ReLU, 
                    net_arch=[dict(pi=[128,128],vf=[128,128])])

    # logs_save - no perturbations | pi=[128,128],vf=[128,128])] | failed_reward=-500.0/-100
    # model = PPO('MlpPolicy', env, learning_rate=0.0001, n_steps=8192, batch_size=128, n_epochs=10, clip_range=0.2, gamma=0.999, gae_lambda=0.90, 
    # use_sde=False, create_eval_env=False, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs_save/")

    # logs_save1 - no perturbations | pi=[128,128],vf=[128,128])] | failed_reward=-500.0/-100 | removed environment make_vec_env seed
    # model = PPO('MlpPolicy', env, learning_rate=0.0001, n_steps=8192, batch_size=128, n_epochs=10, clip_range=0.2, gamma=0.999, gae_lambda=0.90, 
    # use_sde=False, create_eval_env=False, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs_save1/")

    # logs_save2 - no perturbations | pi=[128,128],vf=[128,128])] | failed_reward=-500.0/-100 | removed environment make_vec_env seed | make
    # model = PPO('MlpPolicy', env, learning_rate=0.0001, n_steps=8192, batch_size=128, n_epochs=10, clip_range=0.2, gamma=0.999, gae_lambda=0.90, 
    # use_sde=False, create_eval_env=False, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs_save2/")

    # logs_save3 - 3-100 at 0.9998 perturbations | pi=[128,128],vf=[128,128])] | failed_reward=-500.0/-100 | removed environment make_vec_env seed | added leg spread reward
    # model = PPO('MlpPolicy', env, learning_rate=0.00005, n_steps=8192, batch_size=128, n_epochs=10, clip_range=0.2, gamma=0.999, gae_lambda=0.90, 
    # use_sde=False, create_eval_env=False, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs_save3/")

    # logs_save4 - 3-30 at 0.9998 perturbations | pi=[128,128],vf=[128,128])] | failed_reward=-500.0/-100 | removed environment make_vec_env seed | added leg spread reward
    # model = PPO('MlpPolicy', env, learning_rate=0.000025, n_steps=8192, batch_size=128, n_epochs=5, clip_range=0.2, gamma=0.999, gae_lambda=0.90, 
    # use_sde=False, create_eval_env=False, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs_save4/")

    # logs_save5 - 3-20 at 0.9998 perturbations | pi=[128,128],vf=[128,128])] | failed_reward=-500.0/-100 | removed environment make_vec_env seed | added leg spread reward | removed friction in xml
    # model = PPO('MlpPolicy', env, learning_rate=0.00001, n_steps=8192, batch_size=128, n_epochs=8, clip_range=0.2, gamma=0.999, gae_lambda=0.90, 
    # use_sde=False, create_eval_env=False, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs_save5/")

    # logs_save6 - 3-20 at 0.9998 perturbations | pi=[128,128],vf=[128,128])] | failed_reward=-500.0/-100 | removed environment make_vec_env seed | added leg spread reward | removed friction in xml | e2=-30
    # model = PPO('MlpPolicy', env, learning_rate=0.0001, n_steps=8192, batch_size=128, n_epochs=8, clip_range=0.2, gamma=0.999, gae_lambda=0.90, 
    # use_sde=False, create_eval_env=False, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs_save6/")

    # BEST WORKING VERSION FOR STANDING
    # logs_save7 - 3-100 at 0.9998 perturbations | pi=[128,128],vf=[128,128])] | failed_reward=-500.0/-100 | removed environment make_vec_env seed | added leg spread reward | removed friction in xml | e2=-25
    # model = PPO('MlpPolicy', env, learning_rate=0.0001, n_steps=8192, batch_size=128, n_epochs=8, clip_range=0.2, gamma=0.999, gae_lambda=0.90, 
    # use_sde=False, create_eval_env=False, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs_save7/")

    # logs_walking0 - no perturbations | basic walking rewards
    # model = PPO('MlpPolicy', env, learning_rate=0.0001, n_steps=8192, batch_size=128, n_epochs=8, clip_range=0.2, gamma=0.999, gae_lambda=0.90, 
    # use_sde=False, create_eval_env=False, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs_walking0/")

    # logs_walking1 - no perturbations | basic walking rewards
    model = PPO('MlpPolicy', env, learning_rate=0.0001, n_steps=8192, batch_size=128, n_epochs=8, clip_range=0.2, gamma=0.999, gae_lambda=0.90, 
    use_sde=False, create_eval_env=False, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs_walking1/")







    render_call = RenderCallback(render_freq=8192*5, env=eval_env)
    for i in range(int(500)):
        model.learn(total_timesteps=int(5000000), callback=render_call) #, eval_env=eval_env, eval_freq=1024*5)   #, callback=render_call
        model.save("./logs_walking1/cassie_walking{i}".format(i=i))

