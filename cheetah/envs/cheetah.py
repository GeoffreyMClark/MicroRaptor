from ntpath import join
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
# from gym.envs.registration import register
from gym.envs import register
import math
import os


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}
directory_path = os.getcwd()

class CheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file=directory_path+"/cheetah/envs/mini_cheetah.xml",
        ctrl_cost_weight=0.05,
        contact_cost_weight=2e-1,
        healthy_reward=1.0,
        failed_reward=-100.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.0, 1.0),
        healthy_foot_z=0.2,
        reset_noise_scale=.05,
    ):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._failed_reward = failed_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._healthy_foot_z = healthy_foot_z
        self._reset_noise_scale = reset_noise_scale

        self.prev_action = np.array([0,0,0,0,0,0,0,0,0,0,0,0])

        mujoco_env.MujocoEnv.__init__(self, xml_file, 1)

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        done = self.done
        survival_reward = self._healthy_reward if done == False else self._failed_reward
        reward = survival_reward
        
        observation = self._get_obs()
        self.prev_action = action
        info = {"terminal_observation": done}
        return observation, reward, done, info


    def _get_obs(self):
        sensordata = self.sim.data.sensordata.flat.copy()
        observations = sensordata
        return observations

    def reset_model(self):
        self.prev_action = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(self.model.nv)
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


