from ntpath import join
from random import uniform
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

class CassieEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file=directory_path+"/cassie/envs/cassie.xml",
        ctrl_cost_weight=0.05,
        contact_cost_weight=2e-1,
        healthy_reward=0.0,
        failed_reward=-500.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.3, 1.1),
        healthy_x_range=(-0.3, 0.3),
        healthy_foot_z=0.2,
        healthy_orientation_range=1.25,
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=.05,
        max_episode_steps=300000,
        exclude_current_IMU_from_observation=True,
    ):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._failed_reward = failed_reward
        self._max_episode_steps = max_episode_steps
        self.num_steps = 0
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._healthy_foot_z = healthy_foot_z
        self._healthy_x_range = healthy_x_range
        self._healthy_orientation_range = healthy_orientation_range
        self.left_foot_initial_pose = np.array([0.,0.,0.])
        self.right_foot_initial_pose = np.array([0.,0.,0.])
        self.perturb_force = np.array([0.,0.,0.])
        self.perturb_flag=0

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self.exclude_current_IMU_from_observation = (
            exclude_current_IMU_from_observation
        )

        self.sensor_average=np.array([3.75,15,0,-100.5,-85,0,1.8,-1.5,3.75,15,0,-100.5,-85,0,1.8,-1.5,    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.sensor_scale=np.array([18.75,22.5,65,63.5,55,.5,1.65,1.3,18.75,22.5,65,63.5,55,.5,1.65,1.3,  600,600,600,600,1500,25,40,25,600,600,600,600,1500,25,40,25])
        self.action_scale=np.array([4.5, 4.5, 12.2, 12.2, .9, 4.5, 4.5, 12.2, 12.2, .9])
        # self.prior_scaled_action = np.array([0,0,0,0,0,0,0,0,0,0])
        
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
        left_foot_x, left_foot_y, left_foot_z = self.get_body_com("left-foot")
        right_foot_x, right_foot_y, right_foot_z = self.get_body_com("right-foot")
        quat = self.data.get_body_xquat("cassie-pelvis")
        ori = self.euler_from_quaternion(quat)
        min_z, max_z = self._healthy_z_range
        min_x, max_x = self._healthy_x_range
        max_ori = self._healthy_orientation_range
        max_foot = self._healthy_foot_z
        is_healthy = np.isfinite(state).all()  and  min_z<=state[2]<=max_z  and  left_foot_z<=max_foot  and  right_foot_z<=max_foot  and  min_x<=state[0]<=max_x and -max_ori<=ori[0]-np.pi<=max_ori and -max_ori<=ori[1]<=max_ori and -max_ori<=ori[2]<=max_ori 
        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def step(self, action):
        # scaled_action = np.tanh(action)*self.action_scale
        scaled_action = action

        # if 0 < self.perturb_flag <= 100:
        #     self.perturb_flag = self.perturb_flag + 1
        # elif 10 < self.perturb_flag:
        #     self.perturb_flag=0
        #     self.perturb_force = np.array([0.,0.,0.])

        # if np.random.uniform(0,1) >= 0.9999 and self.perturb_flag==0:
        #     force_p = np.random.uniform(3,20)
        #     r1, r2 = np.random.uniform(0,2*np.pi, 2)
        #     self.perturb_force[0] = force_p * np.sin(r2) * np.cos(r1)
        #     self.perturb_force[1] = force_p * np.sin(r2) * np.sin(r1)
        #     self.perturb_force[2] = force_p * np.cos(r2)
        #     self.perturb_flag = 1

        # self.data.xfrc_applied[1] = np.array([self.perturb_force[0], self.perturb_force[1], self.perturb_force[2], 0., 0., 0.])

        self.do_simulation(scaled_action, self.frame_skip)
        pose = self.get_body_com("cassie-pelvis")[:3].copy()
        pose_vel = self.data.get_body_xvelp("cassie-pelvis")
        quat = self.data.get_body_xquat("cassie-pelvis")
        quat_vel = self.data.get_body_xvelr("cassie-pelvis")
        orientation = self.euler_from_quaternion(quat)
        left_foot_pose = self.get_body_com("left-foot")
        left_foot_vel = self.data.get_body_xvelp("left-foot")
        right_foot_pose = self.get_body_com("right-foot")
        right_foot_vel = self.data.get_body_xvelp("right-foot")
        joint_velocitites = self.sim.data.sensordata.flat.copy()[16:32]
        
        # NEED TO ADD REWARD FOR VELOCITIES (JOINT)
        # NEED TO ADD REWARD FOR LOW JOINT TORQUES
        trunk_position_error = np.abs((pose-[0.0,0.0,0.94])/[0.2, 0.3, 0.3]).sum()
        trunk_stable_position_error = np.abs((pose[0:2]-[0,0])*3).sum()
        trunk_velocity_error = np.abs((pose_vel[0:2])).sum()
        trunk_orientation_error = np.abs((orientation-[np.pi,0,0])/1).sum()
        leftfoot_position_error = np.abs((left_foot_pose-self.left_foot_initial_pose)/0.2).sum()         #left foot 0 pose  [-0.00880596,  0.2150994 ,  0.06848903]
        leftfoot_velocity_error = np.abs(left_foot_vel[0:2]).sum()
        rightfoot_position_error = np.abs((right_foot_pose-self.right_foot_initial_pose)/0.2).sum()       #right foot 0 pose [-0.01198452, -0.17144046,  0.0620562 ]
        rightfoot_velocity_error = np.abs(right_foot_vel[0:2]).sum()
        action_error = (scaled_action**2).sum()
        # action_diff_error = scaled_action - self.prior_scaled_action
        velocity_error = np.abs(joint_velocitites).sum()
        done = self.done

        trunk_position_reward = np.e**(-3*trunk_position_error)
        trunk_orientation_reward = np.e**(-3*trunk_orientation_error)
        trunk_velocity_reward = np.e**(-1*trunk_velocity_error)
        leftfoot_position_reward = np.e**(-3*leftfoot_position_error)
        leftfoot_velocity_reward = np.e**(-3*leftfoot_velocity_error)
        rightfoot_position_reward = np.e**(-3*rightfoot_position_error)
        rightfoot_velocity_reward = np.e**(-3*rightfoot_velocity_error)
        action_reward = np.e**(-0.05*action_error)
        # action_diff_reward = np.e**(-0.05*action_diff_error)
        velocity_reward = np.e**(-.002*velocity_error)
        survival_reward = self._healthy_reward if done == False else self._failed_reward

        goal_reward = .4*trunk_position_reward + .4*trunk_orientation_reward + .2*trunk_velocity_reward
        command_reward = .2*action_reward  + .8*velocity_reward
        foot_reward = .5*leftfoot_velocity_reward + .5*rightfoot_velocity_reward

        reward = survival_reward + .7*goal_reward + .1*command_reward + .2*foot_reward
        
        # self.prior_scaled_action = scaled_action
        observation = self._get_obs()
        done = True if self.num_steps >= self._max_episode_steps else done
        self.num_steps=self.num_steps+1
        info = {"terminal_observation": done}
        return observation, reward, done, info


    def _get_obs(self):
        # position = self.sim.data.qpos.flat.copy()
        # velocity = self.sim.data.qvel.flat.copy()
        # contact_force = self.contact_forces.flat.copy()
        pose = self.data.get_body_xpos("cassie-pelvis")
        pose_vel = self.data.get_body_xvelp("cassie-pelvis")
        quat = self.data.get_body_xquat("cassie-pelvis")
        # orientation = self.euler_from_quaternion(quat)
        orientation_vel = self.data.get_body_xvelr("cassie-pelvis")
        sensordata = self.sim.data.sensordata.flat.copy()

        sensors = sensordata[0:32]

        norm_pose = pose-np.array([0.0,0.0,0.94])
        # norm_orientation = (orientation-np.array([np.pi,0,0]))/np.pi
        norm_quat = self._adjust_circle(quat)/np.pi
        norm_sensors = (sensors-self.sensor_average)/self.sensor_scale

        # observations = np.concatenate((norm_pose, norm_orientation, norm_sensors))
        observations = np.concatenate((norm_pose, pose_vel, norm_quat, orientation_vel, norm_sensors))
        return observations

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        lx, ly, lz = self.get_body_com("left-foot")
        rx, ry, rz = self.get_body_com("right-foot")

        self.left_foot_initial_pose = np.array([lx,ly,0.04])
        self.right_foot_initial_pose = np.array([rx,ry,0.04])

        # qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        zero_pose = np.asarray([  1.00000390e-04,  1.10453331e-05,  9.89611245e-01,  9.99999981e-01,
                            -1.50645122e-06, -1.93453798e-04, -1.53469430e-09,  6.83438358e-02,
                            -8.57605285e-05,  6.21365611e-01,  9.56814499e-01, -1.53944178e-02,
                                2.70039519e-02, -2.89032546e-01, -1.37987287e+00,  3.22283275e-03,
                                1.59371586e+00,  6.91569992e-04, -1.59253155e+00,  1.57410108e+00,
                            -1.69989113e+00, -6.83816099e-02,  8.50431551e-05,  6.21366577e-01,
                                9.55578531e-01, -5.09970882e-02, -6.89472086e-03, -2.90209289e-01,
                            -1.37987225e+00,  3.22283929e-03,  1.59371522e+00,  6.91570380e-04,
                            -1.59253155e+00,  1.57410108e+00, -1.69989113e+00])
        qpos = zero_pose #+ self.np_random.uniform(low=noise_low, high=noise_high, size=zero_pose.shape[0])
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(self.model.nv)
        self.set_state(qpos, qvel)
        self.num_steps=0

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
        if roll_x < 0: roll_x = roll_x + 2*np.pi 
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