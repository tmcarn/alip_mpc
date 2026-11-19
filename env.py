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
        return self.convert_q(q), self.convert_dq(q, dq)  # convert_dq needs the MuJoCo (wxyz) quaternion

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

    def get_hand_contact_forces(self, hand_geoms=None, box_geom="box_geom"):
        '''
        Measured contact force ON the robot from the box at each hand, world frame (summed over
        all contacts between that hand's geom and the box). Zero vector when not touching.
        Returns {hand: (3,)}.
        '''
        if hand_geoms is None:
            hand_geoms = {"right_hand": "right_forearm_geom", "left_hand": "left_forearm_geom"}
        box_id = self.mj_model.geom(box_geom).id
        hand_ids = {h: self.mj_model.geom(g).id for h, g in hand_geoms.items()}

        forces = {h: np.zeros(3) for h in hand_geoms}
        f6 = np.zeros(6)
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            for h, gid in hand_ids.items():
                if {c.geom1, c.geom2} != {gid, box_id}:
                    continue
                # f6[:3] is in the contact frame (row 0 = normal, from geom1 to geom2) and is the
                # force geom1 exerts on geom2 (positive normal pushes them apart)
                mujoco.mj_contactForce(self.mj_model, self.mj_data, i, f6)
                f_world = c.frame.reshape(3, 3).T @ f6[:3]
                forces[h] += f_world if c.geom2 == gid else -f_world  # force on the robot's geom
        return forces

    def get_foot_contact_forces(self, foot_geoms=None, floor_geom="floor"):
        '''
        Measured ground reaction force ON each foot, world frame (summed over all contacts between
        that foot's geom and the floor). Zero vector in swing. Returns {foot: (3,)}.
        '''
        if foot_geoms is None:
            foot_geoms = {"right_foot": "right_foot_geom", "left_foot": "left_foot_geom"}
        floor_id = self.mj_model.geom(floor_geom).id
        foot_ids = {f: self.mj_model.geom(g).id for f, g in foot_geoms.items()}

        forces = {f: np.zeros(3) for f in foot_geoms}
        f6 = np.zeros(6)
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            for f, gid in foot_ids.items():
                if {c.geom1, c.geom2} != {gid, floor_id}:
                    continue
                mujoco.mj_contactForce(self.mj_model, self.mj_data, i, f6)
                f_world = c.frame.reshape(3, 3).T @ f6[:3]
                forces[f] += f_world if c.geom2 == gid else -f_world  # force on the foot
        return forces

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

    # --- offscreen video recording from named MJCF cameras ---
    def start_recording(self, cameras=("sagittal", "top_down"), path_prefix="logs/run",
                        fps=30, width=1280, height=720):
        '''
        Record each camera to <path_prefix>_<camera>.mp4 (one file per camera), at `fps` in SIM time,
        so the video lines up with the logged data. Call record_frame() every sim step and
        stop_recording() at the end. Size must fit <visual><global offwidth/offheight> (1980x1080).
        '''
        import os
        import imageio
        os.makedirs(os.path.dirname(path_prefix) or ".", exist_ok=True)
        self._renderer = mujoco.Renderer(self.mj_model, height, width)
        self._rec_cams = {c: self.mj_model.camera(c).id for c in cameras}
        self._writers = {c: imageio.get_writer(f"{path_prefix}_{c}.mp4", fps=fps, macro_block_size=1)
                         for c in cameras}
        self._rec_paths = [f"{path_prefix}_{c}.mp4" for c in cameras]
        self._frame_dt = 1.0 / fps
        self._next_frame_t = self.mj_data.time

    def record_frame(self):
        '''Grab a frame from every recording camera if it's time for the next one (cheap otherwise).'''
        if getattr(self, "_writers", None) is None:
            return
        if self.mj_data.time < self._next_frame_t - 2 * self._frame_dt:   # sim was reset: resync the clock
            self._next_frame_t = self.mj_data.time
        if self.mj_data.time + 1e-9 < self._next_frame_t:
            return
        self._next_frame_t += self._frame_dt
        for c, cam_id in self._rec_cams.items():
            self._renderer.update_scene(self.mj_data, camera=cam_id)
            if self.viewer is not None:   # debug markers (footstep plan, CoM, frames) live in the viewer's
                self._copy_user_geoms(self.viewer.user_scn, self._renderer.scene)  # scene only
            self._writers[c].append_data(self._renderer.render())

    @staticmethod
    def _copy_user_geoms(src, dst):
        '''Append the debug geoms of one MjvScene (e.g. viewer.user_scn) to another.'''
        for i in range(src.ngeom):
            if dst.ngeom >= dst.maxgeom:
                break
            s = src.geoms[i]
            g = dst.geoms[dst.ngeom]
            mujoco.mjv_initGeom(g, s.type,   # scene stores float32, initGeom wants float64
                                s.size.astype(np.float64), s.pos.astype(np.float64),
                                s.mat.astype(np.float64).ravel(), s.rgba.astype(np.float32))
            g.size[:] = s.size   # initGeom reinterprets capsule sizes ([r, r, half_len] -> [r, r, r]): restore
            dst.ngeom += 1

    def stop_recording(self):
        '''Finish and close the video files. Safe to call when not recording.'''
        if getattr(self, "_writers", None) is None:
            return
        for w in self._writers.values():
            w.close()
        self._renderer.close()
        self._writers = None
        print("saved videos:", ", ".join(self._rec_paths))

    def close(self):
        self.stop_recording()
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

    def convert_dq(self, q, dq):
        '''
        Free-joint velocity convention differs:
            MuJoCo:    [linear vel in WORLD frame, angular vel in LOCAL (base) frame]
            Pinocchio: [linear vel in LOCAL frame, angular vel in LOCAL frame]
        Rotate the linear part into the base frame. Without this, every velocity Pinocchio
        computes (CoM velocity, momentum, Jacobian velocities, Coriolis) is rotated by the
        base yaw -- invisible at zero heading, wrong as soon as the robot turns.
        '''
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q[3:7])  # MuJoCo qpos quaternion (wxyz), base -> world
        dq_conv = dq.copy()
        dq_conv[0:3] = R.reshape(3, 3).T @ dq[0:3]
        return dq_conv