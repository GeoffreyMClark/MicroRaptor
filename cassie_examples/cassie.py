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
        ctrl_cost_weight=0.05,
        contact_cost_weight=2e-1,
        healthy_reward=10.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.5, 1.5),
        healthy_foot_z=0.2,
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_IMU_from_observation=True,
    ):
        utils.EzPickle.__init__(**locals())

        register(id='Cassie-v0', entry_point='cassie_examples.cassie:CassieEnv')

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._healthy_foot_z = healthy_foot_z

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self.exclude_current_IMU_from_observation = (
            exclude_current_IMU_from_observation
        )

        mujoco_env.MujocoEnv.__init__(self, xml_file, 1)

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
        left_foot_z = self.get_body_com("left-foot")[2]
        right_foot_z = self.get_body_com("right-foot")[2]
        min_z, max_z = self._healthy_z_range
        max_foot = self._healthy_foot_z
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z and (left_foot_z <= max_foot and right_foot_z <= max_foot)
        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        pose = self.get_body_com("cassie-pelvis")[:3].copy()
        quat = self.data.get_body_xquat("cassie-pelvis")
        orientation = self.euler_from_quaternion(quat)
        xvel = self.data.get_body_xvelp("cassie-pelvis")
        rvel = self.data.get_body_xvelr("cassie-pelvis")

        total_xvel = np.sum(xvel)
        total_rvel = np.sum(rvel)
        
        done = self.done
        
        orientation_cost = np.sum(np.square(self._adjust_circle(orientation - [np.pi,0,0])*5))
        position_z_cost = np.square(1-pose[2])
        xvel_cost = np.square(total_xvel)
        rvel_cost = np.square(total_rvel)

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost if self.contact_cost>0 else -1000
        healthy_reward = self.healthy_reward
        reward = healthy_reward - position_z_cost - orientation_cost

        # if done:
        #     reward = -5000

        observation = self._get_obs()
        info = {
            # "cost_motion": position_x_cost+position_y_cost,
            # "cost_ctrl": ctrl_cost,
            # "reward_stand": healthy_reward,
            # "x_position": xyz_position_after[0],
            # "y_position": xyz_position_after[1],
            # "distance_from_origin": np.linalg.norm(xyz_position_after, ord=2),
            # "total_velocity": total_xyz_velocity,
            # "y_velocity": y_velocity,
            # "forward_reward": forward_reward,
        }
        return observation, reward, done, info

    # def _get_pose_and_orientation(self):
    #     pose = self.get_body_com("cassie-pelvis")[:3].copy()
    #     quat = self.data.get_body_xquat("cassie-pelvis")
    #     orientation = self.euler_from_quaternion(quat)
    #     return pose, orientation

    def _get_obs(self):
        # position = self.sim.data.qpos.flat.copy()
        # velocity = self.sim.data.qvel.flat.copy()
        # contact_force = self.contact_forces.flat.copy()
        pose = self.get_body_com("cassie-pelvis")[:3].copy()
        quat = self.data.get_body_xquat("cassie-pelvis")
        orientation = self.euler_from_quaternion(quat)
        sensordata = self.sim.data.sensordata.flat.copy()

        if self.exclude_current_IMU_from_observation:
            sensors = sensordata[0:16]
        else:
            sensors = sensordata

        observations = np.concatenate((pose, orientation, sensors))
        # observations = sensors

        return observations

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        # qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qpos = np.asarray([  1.00000390e-04,  1.10453331e-05,  9.89611245e-01,  9.99999981e-01,
                            -1.50645122e-06, -1.93453798e-04, -1.53469430e-09,  6.83438358e-02,
                            -8.57605285e-05,  6.21365611e-01,  9.56814499e-01, -1.53944178e-02,
                                2.70039519e-02, -2.89032546e-01, -1.37987287e+00,  3.22283275e-03,
                                1.59371586e+00,  6.91569992e-04, -1.59253155e+00,  1.57410108e+00,
                            -1.69989113e+00, -6.83816099e-02,  8.50431551e-05,  6.21366577e-01,
                                9.55578531e-01, -5.09970882e-02, -6.89472086e-03, -2.90209289e-01,
                            -1.37987225e+00,  3.22283929e-03,  1.59371522e+00,  6.91570380e-04,
                            -1.59253155e+00,  1.57410108e+00, -1.69989113e+00])
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

    def euler_from_quaternion(self, quat):
        x=quat[0]; y=quat[1]; z=quat[2]; w=quat[3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return np.asarray([roll_x, pitch_y, yaw_z]) # in radians

    def _adjust_circle(self, orientetion):
        temp = np.where(orientetion>(np.pi), orientetion-(2*np.pi), orientetion)
        adjusted = np.where(temp<(-np.pi), temp+(2*np.pi), temp)
        return adjusted 

