from math import gamma
import cv2
import gym
import time
import torch as th
import optuna

import tensorboard

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from cassie.envs.cassie import CassieEnv
from cassie_render_callback import RenderCallback



def optimize_ppo(trial):
    """ Learning hyperparamters we want to optimise"""
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    net_arch = {
        "small": [dict(pi=[128, 128], vf=[128, 128])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
    }[net_arch]
    # activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    # activation_fn = {"tanh": th.nn.Tanh, "relu": th.nn.ReLU, "elu": th.nn.ELU, "leaky_relu": th.nn.LeakyReLU}[activation_fn]

    return {
        'batch_size': trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256]),
        'n_steps': trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
        'gamma': trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]),
        'learning_rate': trial.suggest_loguniform("learning_rate", 1e-5, 1.),
        # 'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'clip_range': trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4]),
        "n_epochs": trial.suggest_categorical("n_epochs", [1, 5, 10]),
        "gae_lambda": trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]),
        "max_grad_norm": trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]),
        # "vf_coef": trial.suggest_uniform("vf_coef", 0, 1),
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=[dict(pi=[512, 256],vf=[512, 256])],
            activation_fn=th.nn.ReLU,
            ortho_init=False,
        ),
    }


def optimize_agent(trial):
    """ Train the model and optimize
        Optuna maximises the negative log likelihood, so we
        need to negate the reward here
    """
    env_id = 'CassieEnv-v0'
    model_params = optimize_ppo(trial)
    env = make_vec_env(env_id, n_envs=8, seed=0, vec_env_cls=DummyVecEnv)
    # model = PPO('MlpPolicy', env, verbose=0, nminibatches=1, **model_params)
    model = PPO('MlpPolicy', env, verbose=1, **model_params, tensorboard_log="./logs_optuna/")
    model.learn(10000000)
    mean_reward, _ = evaluate_policy(model, env_id, n_eval_episodes=1)
    print(model_params)
    return -1 * mean_reward


if __name__ == '__main__':
    study = optuna.create_study()
    try:
        study.optimize(optimize_agent, n_trials=None, n_jobs=1)
    except KeyboardInterrupt:
        print('Interrupted by keyboard.')



