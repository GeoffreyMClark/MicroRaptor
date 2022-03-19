import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.registration import register
import math
import os


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}
directory_path = os.getcwd()

class CassieEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file=directory_path+"/cassie_examples/cassie.xml",
        ctrl_cost_weight=0.5,
        contact_cost_weight=5e-4,
        healthy_reward=10.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.5, 1.1),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_IMU_from_observation=False,
    ):
        utils.EzPickle.__init__(**locals())

        register(id='Cassie-v0', entry_point='cassie_examples.cassie:CassieEnv')

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self.exclude_current_IMU_from_observation = (
            exclude_current_IMU_from_observation
        )

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property
    def healthy_reward(self):
        return (float(self.is_healthy or self._terminate_when_unhealthy) * self._healthy_reward)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(np.square(self.contact_forces))
        return contact_cost

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
        xy_position_before = self.get_body_com("cassie-pelvis")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("cassie-pelvis")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity
        
        done = self.done
        

        forward_cost = np.abs(x_velocity)*10
        sideways_cost = np.abs(y_velocity)*10
        ctrl_cost = self.control_cost(action)*1000
        # contact_cost = self.contact_cost
        healthy_reward = self.healthy_reward
        reward = healthy_reward - forward_cost - sideways_cost - ctrl_cost
        if done:
            reward -= 5000

        observation = self._get_obs()
        info = {
            "cost_motion": forward_cost+sideways_cost,
            "cost_ctrl": ctrl_cost,
            "reward_stand": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            # "forward_reward": forward_reward,
        }
        return observation, reward, done, info

    def _get_obs(self):
        # position = self.sim.data.qpos.flat.copy()
        # velocity = self.sim.data.qvel.flat.copy()
        # contact_force = self.contact_forces.flat.copy()
        sensordata = self.sim.data.sensordata.flat.copy()

        if self.exclude_current_IMU_from_observation:
            sensors = sensordata[0:16]
        else:
            sensors = sensordata

        # observations = np.concatenate((position, velocity, contact_force))
        observations = sensors

        return observations

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)