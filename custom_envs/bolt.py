# Copyright 2021 Geoffrey Clark.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trains bolt robot to run in multiple directions and over obsticals."""

import brax
from brax.envs import env
from brax.physics import bodies
from brax.physics.base import take
import jax
import jax.numpy as jnp


class Bolt(env.Env):

  def __init__(self, **kwargs):
    super().__init__(_SYSTEM_CONFIG, **kwargs)
    body = bodies.Body(self.sys.config)
    body = take(body, body.idx[:-1])  # skip the floor body
    self.mass = body.mass.reshape(-1, 1)
    self.inertia = body.inertia

  def reset(self, rng: jnp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    qp = self.sys.default_qp()
    qp, info = self.sys.step(qp,
                             jax.random.uniform(rng, (self.action_size,)) * .5)
    obs = self._get_obs(qp, info, jnp.zeros(self.action_size))
    reward, done, zero = jnp.zeros(3)
    metrics = {
        'reward_linvel': zero,
        'reward_quadctrl': zero,
        'reward_alive': zero,
        'reward_impact': zero
    }
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jnp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    qp, info = self.sys.step(state.qp, action)
    obs = self._get_obs(qp, info, action)

    pos_before = state.qp.pos[:-1]  # ignore floor at last index
    pos_after = qp.pos[:-1]  # ignore floor at last index
    com_before = jnp.sum(pos_before * self.mass, axis=0) / jnp.sum(self.mass)
    com_after = jnp.sum(pos_after * self.mass, axis=0) / jnp.sum(self.mass)
    lin_vel_cost = 1.25 * (com_after[0] - com_before[0]) / self.sys.config.dt
    quad_ctrl_cost = .01 * jnp.sum(jnp.square(action))
    # can ignore contact cost, see: https://github.com/openai/gym/issues/1541
    quad_impact_cost = 0.0
    alive_bonus = 5.0
    reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus

    done = jnp.where(qp.pos[0, 2] < 0.65, x=1.0, y=0.0)
    done = jnp.where(qp.pos[0, 2] > 2.1, x=1.0, y=done)
    state.metrics.update(
        reward_linvel=lin_vel_cost,
        reward_quadctrl=quad_ctrl_cost,
        reward_alive=alive_bonus,
        reward_impact=quad_impact_cost)

    return state.replace(qp=qp, obs=obs, reward=reward, done=done)

  def _get_obs(self, qp: brax.QP, info: brax.Info,
               action: jnp.ndarray) -> jnp.ndarray:
    """Observe bolt body position, velocities, and angles."""
    # some pre-processing to pull joint angles and velocities
    joint_1d_angle, joint_1d_vel = self.sys.joints[0].angle_vel(qp)
    # joint_2d_angle, joint_2d_vel = self.sys.joints[1].angle_vel(qp)
    # joint_3d_angle, joint_3d_vel = self.sys.joints[2].angle_vel(qp)

    # qpos:
    # Z of the torso (1,)
    # orientation of the torso as quaternion (4,)
    # joint angles (8,)
    qpos = [ qp.pos[0, 2:], qp.rot[0], joint_1d_angle[0]]#, joint_2d_angle[0], joint_2d_angle[1] ]

    # qvel:
    # velocity of the torso (3,)
    # angular velocity of the torso (3,)
    # joint angle velocities (8,)
    qvel = [ qp.vel[0], qp.ang[0], joint_1d_vel[0]]#, joint_2d_vel[0], joint_2d_vel[1] ]

    # actuator forces
    qfrc_actuator = []
    for act in self.sys.actuators:
      torque = take(action, act.act_index)
      torque = torque.reshape(torque.shape[:-2] + (-1,))
      torque *= jnp.repeat(act.strength, act.act_index.shape[-1])
      qfrc_actuator.append(torque)

    # external contact forces:
    # delta velocity (3,), delta ang (3,) * num bodies in the system
    cfrc_ext = [info.contact.vel, info.contact.ang]
    # flatten bottom dimension
    cfrc_ext = [x.reshape(x.shape[:-2] + (-1,)) for x in cfrc_ext]

    # center of mass obs:
    body_pos = qp.pos[:-1]  # ignore floor at last index
    body_vel = qp.vel[:-1]  # ignore floor at last index

    com_vec = jnp.sum(body_pos * self.mass, axis=0) / jnp.sum(self.mass)
    com_vel = body_vel * self.mass / jnp.sum(self.mass)

    def v_outer(a):
      return jnp.outer(a, a)

    def v_cross(a, b):
      return jnp.cross(a, b)

    v_outer = jax.vmap(v_outer, in_axes=[0])
    v_cross = jax.vmap(v_cross, in_axes=[0, 0])

    disp_vec = body_pos - com_vec
    # com_inert = self.inertia + self.mass.reshape(
    #     (11, 1, 1)) * ((jnp.linalg.norm(disp_vec, axis=1)**2.).reshape(
    #         (11, 1, 1)) * jnp.stack([jnp.eye(3)] * 11) - v_outer(disp_vec))
    com_inert = self.inertia + self.mass.reshape(
        (7, 1, 1)) * ((jnp.linalg.norm(disp_vec, axis=1)**2.).reshape(
            (7, 1, 1)) * jnp.stack([jnp.eye(3)] * 7) - v_outer(disp_vec))

    cinert = [com_inert.reshape(-1)]

    square_disp = (1e-7 + (jnp.linalg.norm(disp_vec, axis=1)**2.)).reshape(
        (7, 1))
    com_angular_vel = (v_cross(disp_vec, body_vel) / square_disp)
    cvel = [com_vel.reshape(-1), com_angular_vel.reshape(-1)]

    return jnp.concatenate(qpos + qvel + cinert + cvel + qfrc_actuator +
                           cfrc_ext)

_SYSTEM_CONFIG = """
bodies {
  name: "base_link"
  colliders {
      capsule {
          radius: 0.09
          length: 0.32
      }
      rotation {
          x: -90.0
      }
      position {
      }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 6.6161942
}
bodies {
  name: "FR_SHOULDER"
  colliders {
      capsule {
          radius: 0.04,
          length: 0.06
      }
      rotation {
          y: 90.0
      }
      position {
          y: -0.13
          z: -0.07
      }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.751751
}
bodies {
  name: "FL_SHOULDER"
  colliders {
      capsule {
          radius: 0.04,
          length: 0.06
      }
      rotation {
          y: 90.0
      }
      position {
          y: 0.13
          z: -0.07
      }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.751751
}
bodies {
  name: "FR_UPPER_LEG"
  colliders {
      capsule {
          radius: 0.018
          length: 0.3
      }
      position {
          y: -0.18
          z: -0.200
          x: -0.001
      }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.751751
}
bodies {
  name: "FL_UPPER_LEG"
  colliders {
    position {
        y: 0.18
        z: -0.200
        x: -0.001
    }
    rotation {
  
    }
    capsule {
      radius: 0.018
      length: 0.3
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.751751
}
bodies {
  name: "FR_LOWER_LEG"
  colliders {
      capsule {
          radius: 0.015
          length: 0.3
      }
      position {
        x: 0  
        y: -0.21
        z: -0.45
      }
      rotation{
        
      }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.5228419
}
bodies {
  name: "FL_LOWER_LEG"
  colliders {
    position {
      x: 0 
      y: 0.21
      z: -0.45
    }
    rotation {
      
    }
    capsule {
      radius: 0.018
      length: 0.3
    }
  }

  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.5228419
}
bodies {
  name: "floor"
  colliders {
    plane {
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    all: true
  }
}
joints {
  name: "l_base_shoulder_joint"
  stiffness: 8000.0
  parent: "base_link"
  child: "FL_SHOULDER"
  parent_offset {
 
  }
  rotation {
  }
  
  angle_limit {
      min: -360.0
      max: 360.0
  }
  limit_strength: 2000.0
  angular_damping: 20.0
}
joints {
  name: "r_base_shoulder_joint"
  stiffness: 8000.0
  parent: "base_link"
  child: "FR_SHOULDER"
  parent_offset {

  }
  rotation {
  }
  
  angle_limit {
      min: -360.0
      max: 360.0
  }
  limit_strength: 2000.0
  angular_damping: 20.0
}
joints {
  name: "l_shoulder_upperleg_joint"
  stiffness: 8000.0
  parent: "FL_SHOULDER"
  child: "FL_UPPER_LEG"
  parent_offset {
    
  }
  child_offset{
  }
  rotation{
    x: 90.0
  }
  angle_limit {
    min: -360.0
    max: 360.0
  }
  
  limit_strength: 2000.0
  angular_damping: 20.0
}
joints {
  name: "r_shoulder_upperleg_joint"
  stiffness: 8000.0
  parent: "FR_SHOULDER"
  child: "FR_UPPER_LEG"
  parent_offset {
 
  }
  rotation {
    x: 90.0
  }
  angle_limit {
    min: -360.0
    max: 360.0
  }
  limit_strength: 300.0
  angular_damping: 20.0
}
joints {
  name: "l_knee_joint"
  stiffness: 8000.0
  parent: "FL_UPPER_LEG"
  child: "FL_LOWER_LEG"
  parent_offset {

  }
  child_offset {
      
  }
  rotation {
    x: 90.0
  }
  angle_limit {
    min: -360.0
    max: 360.0
  }
  angular_damping: 20.0
}
joints {
  name: "r_knee_joint"
  stiffness: 8000.0
  parent: "FR_UPPER_LEG"
  child: "FR_LOWER_LEG"
  parent_offset {

  }
  child_offset {
  }
  rotation {
    x: 90.0
  }
  angle_limit {
    min: -360.0
    max: 360.0
  }
  angular_damping: 20.0
}
actuators {
  name: "l_base_shoulder_joint"
  joint: "l_base_shoulder_joint"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "l_shoulder_upperleg_joint"
  joint: "l_shoulder_upperleg_joint"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "l_knee_joint"
  joint: "l_knee_joint"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "r_base_shoulder_joint"
  joint: "r_base_shoulder_joint"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "r_shoulder_upperleg_joint"
  joint: "r_shoulder_upperleg_joint"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "r_knee_joint"
  joint: "r_knee_joint"
  strength: 300.0
  torque {
  }
}


defaults {
    angles {
        name: "l_knee_joint"
        angle {x: 0 y: 0 z: 0}
    }
    angles{
        name: "r_knee_joint"
        angle {x: 0 y: 0 z: 0}
    }
}

collide_include {
  first: "floor"
  second: "FL_LOWER_LEG"
}
collide_include {
  first: "floor"
  second: "FR_LOWER_LEG"
}
friction: -0.6
gravity {
  z: -9.81
}
angular_damping: -0.05
baumgarte_erp: 0.1
dt: 0.015
substeps: 8
"""



