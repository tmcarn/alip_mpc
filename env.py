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

        self.torso = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")

        # Unactuated box (free body). It adds 7 qpos / 6 qvel entries that the
        # robot-only Pinocchio model doesn't know about, so keep them separate.
        self.box = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "box")
        box_jnt = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "box_free")
        qa, va = self.mj_model.jnt_qposadr[box_jnt], self.mj_model.jnt_dofadr[box_jnt]
        self.box_qpos_idx = np.arange(qa, qa + 7)
        self.box_qvel_idx = np.arange(va, va + 6)
        self.robot_qpos_idx = np.setdiff1d(np.arange(self.mj_model.nq), self.box_qpos_idx)
        self.robot_qvel_idx = np.setdiff1d(np.arange(self.mj_model.nv), self.box_qvel_idx)

        # The controller outputs tau in joint (dof) order, but MuJoCo's ctrl is in
        # <actuator> order, which differs for the humanoid (legs listed before
        # arms). Map each actuator to its joint's index in the robot's tau vector.
        robot_act_dofs = list(self.robot_qvel_idx[6:])  # actuated dofs, skip floating base
        self.act_to_tau = np.array([
            robot_act_dofs.index(self.mj_model.jnt_dofadr[self.mj_model.actuator_trnid[a, 0]])
            for a in range(self.mj_model.nu)
        ])
        # Per-joint torque limits (tau order), from each actuator's forcerange/ctrlrange.
        self.tau_limits = np.full(len(robot_act_dofs), np.inf)
        for a, k in enumerate(self.act_to_tau):
            for limited, rng in ((self.mj_model.actuator_forcelimited[a], self.mj_model.actuator_forcerange[a]),
                                 (self.mj_model.actuator_ctrllimited[a], self.mj_model.actuator_ctrlrange[a])):
                if limited:
                    self.tau_limits[k] = min(self.tau_limits[k], np.abs(rng).min())
        self.tau_limits[np.isinf(self.tau_limits)] = ACTUATOR_LIMIT

    def reset(self):
        if self.mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
        else:
            mujoco.mj_resetData(self.mj_model, self.mj_data)
        # The robot's keyframe doesn't cover the box, so MuJoCo zero-fills it
        # (box at the origin). Restore the box's default pose from box.xml.
        self.mj_data.qpos[self.box_qpos_idx] = self.mj_model.qpos0[self.box_qpos_idx]
        self.mj_data.qvel[self.box_qvel_idx] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)
        return self.get_joint_state()

    def get_joint_state(self):
        '''Robot-only (q, dq), matching the Pinocchio model (box excluded).'''
        q = self.mj_data.qpos[self.robot_qpos_idx].copy()
        dq = self.mj_data.qvel[self.robot_qvel_idx].copy()
        return self.convert_q(q), dq

    def set_joint_angles(self, angles_deg):
        '''Set hinge joints {name: degrees} in the sim state (call after reset) and return the new (q, dq).'''
        for name, deg in angles_deg.items():
            self.mj_data.qpos[self.mj_model.jnt_qposadr[self.mj_model.joint(name).id]] = np.radians(deg)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        return self.get_joint_state()

    def robot_q_index(self, joint_name):
        '''Index of a hinge joint's angle in the robot-only q from get_joint_state().'''
        adr = self.mj_model.jnt_qposadr[self.mj_model.joint(joint_name).id]
        return int(np.where(self.robot_qpos_idx == adr)[0][0])

    def get_box_state(self):
        '''Box pose/velocity: pos (3,), quat wxyz (4,), linear vel (3,), angular vel (3,).'''
        qb = self.mj_data.qpos[self.box_qpos_idx]
        vb = self.mj_data.qvel[self.box_qvel_idx]
        return qb[:3].copy(), qb[3:7].copy(), vb[:3].copy(), vb[3:].copy()

    def step(self, tau):
        # tau: actuated joints only, in joint (Pinocchio dof) order
        self.mj_data.ctrl[:] = np.asarray(tau)[self.act_to_tau]
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