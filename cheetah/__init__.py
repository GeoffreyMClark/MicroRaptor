from gym.envs.registration import register

register(id='CheetahEnv-v0', entry_point='cheetah.envs:CheetahEnv', max_episode_steps=100000)