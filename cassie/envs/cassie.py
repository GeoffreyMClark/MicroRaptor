from ntpath import join
from random import uniform
import numpy as np
from scipy.stats import vonmises
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
        healthy_reward=1.,
        failed_reward=-500.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.75, 1.05),
        healthy_x_range=(-0.2, 100),
        healthy_y_range=(-0.2, 0.2),
        healthy_foot_z=0.2,
        healthy_orientation_range=1.25,
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=.05,
        max_episode_steps=8192*1.5,
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
        self._healthy_y_range = healthy_y_range
        self._healthy_orientation_range = healthy_orientation_range
        self.left_foot_initial_pose = np.array([0.,0.,0.])
        self.right_foot_initial_pose = np.array([0.,0.,0.])
        self.perturb_force = np.array([0.,0.,0.])
        self.perturb_flag=0
        self.goal_velocity_min = 0
        self.goal_velocity_max = 1.5
        self.goal_velocity=0
        self.phi=0
        self.phi_dt=0.001
        self.cycle1_stance=np.linspace(0,0.999,1000)
        self.cycle1_swing=np.linspace(0,0.999,1000)
        self.cycle2_stance=np.linspace(0,0.999,1000)
        self.cycle2_swing=np.linspace(0,0.999,1000)
        self.r=0.6
        self.r_min=0.4
        self.r_max=0.7
        self.cycle_time=1.
        self.cycle_time_min=0.7
        self.cycle_time_max=1.4
        self.stand_flag=0


        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self.exclude_current_IMU_from_observation = (
            exclude_current_IMU_from_observation
        )

        self.sensor_average=np.array([3.75,15,0,-100.5,-85,0,1.8,-1.5,3.75,15,0,-100.5,-85,0,1.8,-1.5,    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,                                0,0,0,0,0,0,0,0,0,0])
        self.sensor_scale=np.array([18.75,22.5,65,63.5,55,.5,1.65,1.3,18.75,22.5,65,63.5,55,.5,1.65,1.3,  600,600,600,600,1500,25,40,25,600,600,600,600,1500,25,40,25,    4.5, 4.5, 12.2, 12.2, .9, 4.5, 4.5, 12.2, 12.2, .9])
        self.action_scale=np.array([4.5, 4.5, 12.2, 12.2, .9, 4.5, 4.5, 12.2, 12.2, .9])
        self.prior_scaled_action = np.array([0,0,0,0,0,0,0,0,0,0])
        
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
        min_y, max_y = self._healthy_y_range
        max_ori = self._healthy_orientation_range
        max_foot = self._healthy_foot_z
        is_healthy = np.isfinite(state).all()  and  min_z<=state[2]<=max_z  and min_x<=state[0]<=max_x and min_y<=state[1]<=max_y   and -max_ori<=ori[0]-np.pi<=max_ori and -max_ori<=ori[1]<=max_ori and -max_ori<=ori[2]<=max_ori 
        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def step(self, action):
        # scale action space with tanh function
        scaled_action = np.tanh(action)*self.action_scale

        # Add perturbations with random direction
        pert_t_steps=100; pert_min=0; pert_max=5; pert_prob=.0002
        if 0 < self.perturb_flag <= pert_t_steps:
            self.perturb_flag = self.perturb_flag + 1
        elif self.perturb_flag > pert_t_steps:
            self.perturb_flag=0
            self.perturb_force = np.array([0.,0.,0.])
        if np.random.uniform(0,1) <= pert_prob and self.perturb_flag==0:
            force_p = np.random.uniform(pert_min,pert_max)
            r1, r2 = np.random.uniform(0,2*np.pi, 2)
            self.perturb_force[0] = force_p * np.sin(r2) * np.cos(r1)
            self.perturb_force[1] = force_p * np.sin(r2) * np.sin(r1)
            self.perturb_force[2] = force_p * np.cos(r2)
            self.perturb_flag = 1
        self.data.xfrc_applied[1] = np.array([self.perturb_force[0], self.perturb_force[1], self.perturb_force[2], 0., 0., 0.])

        # Run simulation step
        self.do_simulation(scaled_action, self.frame_skip)

        # Get simulation states
        pose = self.get_body_com("cassie-pelvis")[:3].copy()
        trunk_vel = self.data.get_body_xvelp("cassie-pelvis")
        quat = self.data.get_body_xquat("cassie-pelvis")
        contact_force = self.contact_forces.flat.copy()
        left_foot_pose = self.get_body_com("left-foot")
        left_foot_vel = self.data.get_body_xvelp("left-foot")
        right_foot_pose = self.get_body_com("right-foot")
        right_foot_vel = self.data.get_body_xvelp("right-foot")
        action_diff = np.abs(scaled_action - self.prior_scaled_action).sum()
        scaled_torque = np.abs(scaled_action).sum() 

        # Create gait cycle 
        phi_sel = int((self.phi%1)*(1/self.dt))
        I1_stance = self.cycle1_stance[phi_sel]
        I1_swing = self.cycle1_swing[phi_sel]
        I2_stance = self.cycle2_stance[phi_sel]
        I2_swing = self.cycle2_swing[phi_sel]

        # Calculate survival reward
        done = self.done
        if done == True:
            survival_reward = self._failed_reward if np.abs(pose[0]) > self._healthy_x_range[1] or np.abs(pose[1]) > self._healthy_x_range[1] else self._failed_reward/5
        else:
            survival_reward = self._healthy_reward

        # End simulation if max time steps has been reached with no negative reward
        done = True if self.num_steps >= self._max_episode_steps else done

        # Calculate reward functions
        e1=-30; e2=-25; e3=-15; e4=-3; e5=-.1; e6=-5; e7=-8; e8=-100
        goal_vel_x=self.goal_velocity; goal_vel_y=0.; 
        goal_quat=[1.,0.,0.,0.]
        R_biped = I1_stance*np.e**(e1*np.abs(left_foot_vel).sum()) + I2_stance*np.e**(e1*np.abs(right_foot_vel).sum())
        R_cmd = np.e**(e2*np.abs(trunk_vel[0]-goal_vel_x)) + np.e**(e2*np.abs(trunk_vel[1]-goal_vel_y)) + np.e**(e3*(1-np.dot(quat, goal_quat)**2))
        R_smooth = np.e**(e4*action_diff) + np.e**(e5*scaled_torque) 
        R_standing = np.e**(e6*(np.abs(left_foot_pose[0]-right_foot_pose[0])+np.abs(left_foot_pose[2]-right_foot_pose[2]))) + np.e**(e7*np.abs(.3836-(left_foot_pose[1]-right_foot_pose[1]))) + np.e**(e8*action_diff)  #
        reward = survival_reward + 0.5*0.5*R_biped + 0.3*0.3333*R_cmd + 0.1*0.5*R_smooth + 0.1*0.3333*R_standing*self.stand_flag

        # Save important parameters
        self.prior_scaled_action = scaled_action
        self.num_steps=self.num_steps+1
        self.phi = self.phi+self.phi_dt
        self.phi = self.phi-1 if self.phi>=1 else self.phi

        # Get observation and info then return
        observation = self._get_obs()
        info = {"terminal_observation": done,
                "time_step": self.num_steps}
        return observation, reward, done, info


    def _get_obs(self):
        pose = self.data.get_body_xpos("cassie-pelvis")
        pose_vel = self.data.get_body_xvelp("cassie-pelvis")
        quat = self.data.get_body_xquat("cassie-pelvis")
        orientation_vel = self.data.get_body_xvelr("cassie-pelvis")
        sensordata = self.sim.data.sensordata.flat.copy()
        sensors = sensordata
        norm_pose = pose[2]-np.array([0.94])
        norm_sensors = (sensors-self.sensor_average)/self.sensor_scale
        control = np.concatenate((  np.array([np.sin(self.phi/self.cycle_time*2*np.pi)]), 
                                    np.array([np.cos(self.phi/self.cycle_time*2*np.pi)]), 
                                    np.array([self.cycle_time]),
                                    np.array([self.goal_velocity]), 
                                    np.array([self.r])))
        print(control)
        observations = np.concatenate((norm_pose, pose_vel, quat, orientation_vel, norm_sensors, control))
        return observations


    def reset_model(self):
        self.goal_velocity = np.random.uniform(self.goal_velocity_min, self.goal_velocity_max)
        self.goal_velocity = 0 if self.goal_velocity <= 0.1 else self.goal_velocity
        self.stand_flag = 1 if self.goal_velocity==0 else 0
        self.phi=0
        self.r = np.random.uniform(self.r_min, self.r_max)
        self.cycle1_stance, self.cycle1_swing, self.cycle2_stance, self.cycle2_swing = self.generate_cycles(self.r)
        self.cycle_time=1#np.random.uniform()

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

    def generate_cycles(self, r):
        phi = np.linspace(0,1, 1000)
        # r=.6; 
        cycle_offset=.5
        a=0; b=a+r; k=100
        ca_right = -vonmises.pdf(phi, k, loc=a, scale=1/(2*np.pi))
        cb_right = vonmises.pdf(phi, k, loc=b, scale=1/(2*np.pi))
        c_sum_right = np.cumsum(ca_right+cb_right)
        c_scale_right = -(c_sum_right.max()-c_sum_right.min())/2
        c_final_right = ((c_sum_right/c_scale_right))/2
        c_final_right = c_final_right-c_final_right.min()
        ca_left = -vonmises.pdf(phi+cycle_offset, k, loc=a, scale=1/(2*np.pi))
        cb_left = vonmises.pdf(phi+cycle_offset, k, loc=b, scale=1/(2*np.pi))
        c_sum_left = np.cumsum(ca_left+cb_left)
        c_scale_left = -(c_sum_left.max()-c_sum_left.min())/2
        c_final_left = ((c_sum_left/c_scale_left))/2
        c_final_left = c_final_left-c_final_left.min()
        c_final_right_stance = c_final_right
        c_final_right_swing  = (-(c_final_right-0.5))+0.5
        c_final_left_stance = c_final_left
        c_final_left_swing  = (-(c_final_left-0.5))+0.5
        return c_final_right_stance, c_final_right_swing, c_final_left_stance, c_final_left_swing

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