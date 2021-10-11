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
    joint_2d_angle, joint_2d_vel = self.sys.joints[1].angle_vel(qp)
    # joint_3d_angle, joint_3d_vel = self.sys.joints[2].angle_vel(qp)

    # qpos:
    # Z of the torso (1,)
    # orientation of the torso as quaternion (4,)
    # joint angles (8,)
    qpos = [ qp.pos[0, 2:], qp.rot[0], joint_1d_angle[0], joint_2d_angle[0], joint_2d_angle[1] ]

    # qvel:
    # velocity of the torso (3,)
    # angular velocity of the torso (3,)
    # joint angle velocities (8,)
    qvel = [ qp.vel[0], qp.ang[0], joint_1d_vel[0], joint_2d_vel[0], joint_2d_vel[1] ]

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
        (5, 1, 1)) * ((jnp.linalg.norm(disp_vec, axis=1)**2.).reshape(
            (5, 1, 1)) * jnp.stack([jnp.eye(3)] * 5) - v_outer(disp_vec))

    cinert = [com_inert.reshape(-1)]

    square_disp = (1e-7 + (jnp.linalg.norm(disp_vec, axis=1)**2.)).reshape(
        (5, 1))
    com_angular_vel = (v_cross(disp_vec, body_vel) / square_disp)
    cvel = [com_vel.reshape(-1), com_angular_vel.reshape(-1)]

    return jnp.concatenate(qpos + qvel + cinert + cvel + qfrc_actuator +
                           cfrc_ext)


_SYSTEM_CONFIG = """
bodies {
  name: "pelvis"
  colliders {
    position {
      x: 0.000
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.058
      length: 0.170
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
  name: "right_thigh"
  colliders {
    position {
      y: 0.000
      z: -0.100
    }
    rotation {
      x: -178.31532
    }
    capsule {
      radius: 0.030
      length: 0.200
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
  name: "right_shin"
  colliders {
    position {
      z: 0.100
      y: -0.00
    }
    rotation {
      x: -180.0
    }
    capsule {
      radius: 0.02
      length: 0.200
      end: -1
    }
  }
  colliders {
    position {
      z: -0.00
    }
    capsule {
      radius: 0.025
      length: 0.025
      end: 1
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
  name: "left_thigh"
  colliders {
    position {
      y: -0.000
      z: -0.100
    }
    rotation {
      x: 178.31532
    }
    capsule {
      radius: 0.030
      length: 0.200
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
  name: "left_shin"
  colliders {
    position {
      z: -0.15
    }
    rotation {
      x: -180.0
    }
    capsule {
      radius: 0.02
      length: 0.398
      end: -1
    }
  }
  colliders {
    position {
      z: -0.35
    }
    capsule {
      radius: 0.025
      length: 0.025
      end: 1
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
  frozen { all: true }
}
joints {
  name: "right_hip_x"
  stiffness: 8000.0
  parent: "pelvis"
  child: "right_thigh"
  parent_offset {
    y: -0.1
    z: -0.04
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 20.0
  limit_strength: 2000.0
  angle_limit {
    min: -10.0
    max: 10.0
  }
  angle_limit {
    min: -30.0
    max: 70.0
  }
  angle_limit {
    min: -10.0
    max: 10.0
  }
}
joints {
  name: "right_knee"
  stiffness: 15000.0
  parent: "right_thigh"
  child: "right_shin"
  parent_offset {
    y: 0.01
    z: -0.383
  }
  child_offset {
    z: 0.02
  }
  rotation {
    z: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -160.0
    max: -2.0
  }
}
joints {
  name: "left_hip_x"
  stiffness: 8000.0
  parent: "pelvis"
  child: "left_thigh"
  parent_offset {
    y: 0.1
    z: -0.04
  }
  child_offset {
  }
  angular_damping: 20.0
  limit_strength: 2000.0
  angle_limit {
    min: -10.0
    max: 10.0
  }
  angle_limit {
    min: -30.0
    max: 70.0
  }
  angle_limit {
    min: -10.0
    max: 10.0
  }
}
joints {
  name: "left_knee"
  stiffness: 15000.0
  parent: "left_thigh"
  child: "left_shin"
  parent_offset {
    y: -0.01
    z: -0.383
  }
  child_offset {
    z: 0.02
  }
  rotation {
    z: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -160.0
    max: -2.0
  }
}
actuators {
  name: "right_hip_x"
  joint: "right_hip_x"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "right_knee"
  joint: "right_knee"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "left_hip_x"
  joint: "left_hip_x"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "left_knee"
  joint: "left_knee"
  strength: 300.0
  torque {
  }
}
collide_include {
  first: "floor"
  second: "left_shin"
}
collide_include {
  first: "floor"
  second: "right_shin"
}
defaults {
  angles {
    name: "left_knee"
    angle { x: 45 y: 0 z: 0 }
  }
  angles {
    name: "right_knee"
    angle { x: 45 y: 0 z: 0 }
  }
}
friction: 1.0
gravity {
  z: -9.81
}
angular_damping: -0.05
baumgarte_erp: 0.1
dt: 0.015
substeps: 8
"""