from gym.envs.registration import register

register(id='CassieEnv-v0', entry_point='cassie.envs:CassieEnv', max_episode_steps=100000)