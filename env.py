import mujoco
import mujoco.viewer
import numpy as np
import time
from constants import *


class BipedEnv:
    '''
    MuJoCo environment wrapper for the biped model. This class handles:
        - MuJoCo model and data management
        - State retrieval (joint states)
        - Stepping the simulation with given control inputs
        - Rendering (using MuJoCo's built-in viewer)
    '''

    def __init__(self, xml_path: str):
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.dt = self.mj_model.opt.timestep  # control timestep

        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        self.foot_body = {
            "right_foot": mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_foot"),
            "left_foot":  mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_foot"),
        }

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        return self.get_joint_state()

    def get_joint_state(self):
        q = self.mj_data.qpos.copy()
        dq = self.mj_data.qvel.copy()
        return self.convert_q(q), dq

    def step(self, tau):
        # tau shape (10,) — actuated joints only
        self.mj_data.ctrl[:] = tau
        mujoco.mj_step(self.mj_model, self.mj_data)
        return self.get_joint_state()

    def render(self):
        '''Sync the viewer to the current sim state.'''
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def convert_q(self, q):
        '''
        Swaps Quaternion in the Configuration Vector
        '''
        q_conv = q.copy()
        q_conv[3:7] = np.array([q[4], q[5], q[6], q[3]])  # wxyz -> xyzw

        return q_conv