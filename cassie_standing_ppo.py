from cv2 import determinant
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from cassie.envs.cassie import CassieEnv


# env = CassieEnv()
# model = PPO('MlpPolicy', env, verbose=1)
# for i in range(int(1e4)):
#     model.learn(total_timesteps=int(2e5))
#     model.save(f"dqn_cassie_{i}_larger_healthy_reward")
#     mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
#     print(mean_reward, std_reward)
# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, done, info = env.step(action)
#     env.render()


if __name__ == '__main__':
    env_id = 'CassieEnv-v0'
    num_cpu = 7  # Number of processes to use
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=DummyVecEnv)

    model = PPO('MlpPolicy', env, verbose=1)
    for i in range(int(1e6)):
        model.learn(total_timesteps=int(2e6))

        obs = env.reset()
        for _ in range(1000):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()