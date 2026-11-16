from pyexpat import model
from typing import Optional

import pinocchio as pin
import numpy as np
from qpsolvers import solve_qp
from constants import *
from dataclasses import dataclass, field


@dataclass
class WBCReference:
    '''Per-tick task-space references for the WBC (world frame).'''
    # Locomotion task references
    swing_pos: np.ndarray                                                 # (3,)
    swing_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))    # (3,)
    swing_rot: np.ndarray = field(default_factory=lambda: np.eye(3))      # (3,3)
    swing_omega: np.ndarray = field(default_factory=lambda: np.zeros(3))  # (3,)
    torso_rot: np.ndarray = field(default_factory=lambda: np.eye(3))      # (3,3)
    torso_omega: np.ndarray = field(default_factory=lambda: np.zeros(3))  # (3,)

    # Manipulation task references
    # None -> that hand's task is off and its arm holds the posture setpoint
    right_hand_pos:   Optional[np.ndarray] = None   # (3,)
    right_hand_vel:   Optional[np.ndarray] = None   # (3,)  None -> zero
    right_hand_force: Optional[np.ndarray] = None   # (3,) force ON the robot from the box (= +n·F/2), step 2

    left_hand_pos:   Optional[np.ndarray] = None   # (3,)
    left_hand_vel:   Optional[np.ndarray] = None   # (3,)
    left_hand_force: Optional[np.ndarray] = None   # (3,) force ON the robot from the box (= +n·F/2), step 2





class WholeBodyController:

    def __init__(self, model, data, tau_limits=None, q_ref=None):
        self.model = model
        self.data = data
        self.feet = ["right_foot", "left_foot"]

        self.nv = self.model.nv          # total dofs (6 floating base + actuated)
        self.n_vars = self.nv + 3        # decision vars: qdd plus stance contact force (x, y, z)

        # per-joint torque limits in dof order (falls back to the scalar limit)
        self.tau_limits = np.full(self.nv - 6, ACTUATOR_LIMIT, dtype=float) if tau_limits is None else np.asarray(tau_limits, dtype=float)

        self.swing_id = None
        self.stance_id = None
        # BODY type: the humanoid also has sites named like the bodies (e.g. "right_foot")
        self.torso_id = self.model.getFrameId("torso", pin.FrameType.BODY)

        # Arm joints (shoulder/elbow) aren't used by the walking tasks, so hold them
        # at fixed setpoints (the arm entries of q_ref) with a joint-space PD.
        self.q_ref = pin.neutral(self.model) if q_ref is None else np.asarray(q_ref).copy()
        self.arm_v_idx, self.arm_q_idx = [], []
        for j in range(2, self.model.njoints):  # skip universe and floating base
            if "shoulder" in self.model.names[j] or "elbow" in self.model.names[j]:
                jm = self.model.joints[j]
                self.arm_v_idx += list(range(jm.idx_v, jm.idx_v + jm.nv))
                self.arm_q_idx += list(range(jm.idx_q, jm.idx_q + jm.nq))
        self.S_arm = np.eye(self.nv)[self.arm_v_idx]  # selects arm dofs

        # Hand frames. MJCF sites are loaded by Pinocchio as OP_FRAME (there is no FrameType.SITE)
        self.right_hand_id = self.model.getFrameId("right_hand", pin.FrameType.OP_FRAME)
        self.left_hand_id = self.model.getFrameId("left_hand", pin.FrameType.OP_FRAME)

        self.com_desired_height = Z_H

        self.mu = MU

        # PD Controller gains
        # swing foot: move fast, light damping
        self.Kp_swing, self.Kd_swing = 900.0, 30.0
        # torso orientation: stiff, well-damped
        self.Kp_torso, self.Kd_torso = 100.0, 20.0
        # CoM height: moderate
        self.Kp_com, self.Kd_com = 100.0, 20.0
        # Swing Foot Orientation
        self.Kp_swing_rot, self.Kd_swing_rot = 100.0, 20.0
        # Arm posture hold
        self.Kp_arm, self.Kd_arm = 100.0, 20.0
        # Hand position (critically damped: Kd = 2 sqrt(Kp))
        self.Kp_hand, self.Kd_hand = 400.0, 40.0

        # QP task weights
        self.w_swing, self.w_torso, self.w_com, self.w_swing_rot, self.w_reg = 200.0, 1.0, 1.0, 1.0, 0.01
        self.w_arm = 1.0         # arm posture hold
        self.w_hand = 20.0       # keep well below w_swing; >> w_torso makes the torso lean to reach

    def compute_jacobians(self, q, dq, stance_foot):
        '''Update the task Jacobians (and Jdot terms) stored on self for this tick.'''
        # Get frame IDs
        self.stance_id = self.model.getFrameId(stance_foot, pin.FrameType.BODY)
        self.swing_id = self.model.getFrameId("left_foot" if stance_foot == "right_foot" else "right_foot", pin.FrameType.BODY)

        pin.computeJointJacobians(self.model, self.data, q)
        pin.computeJointJacobiansTimeVariation(self.model, self.data, q, dq)

        frame = pin.LOCAL_WORLD_ALIGNED

        # stance foot (linear rows 0:3)
        self.J_stance = pin.getFrameJacobian(self.model, self.data, self.stance_id, frame)[:3, :]
        self.Jdot_stance = pin.getFrameJacobianTimeVariation(self.model, self.data, self.stance_id, frame)[:3, :]

        # swing foot: position (rows 0:3) and orientation (rows 3:6)
        J_sw = pin.getFrameJacobian(self.model, self.data, self.swing_id, frame)
        Jdot_sw = pin.getFrameJacobianTimeVariation(self.model, self.data, self.swing_id, frame)
        self.J_swing, self.J_swing_rot = J_sw[:3, :], J_sw[3:, :]
        self.Jdot_swing, self.Jdot_swing_rot = Jdot_sw[:3, :], Jdot_sw[3:, :]

        # hands: position (rows 0:3) only, orientation is not used
        J_rh = pin.getFrameJacobian(self.model, self.data, self.right_hand_id, frame)
        Jdot_rh = pin.getFrameJacobianTimeVariation(self.model, self.data, self.right_hand_id, frame)
        self.J_right_hand, self.Jdot_right_hand = J_rh[:3, :], Jdot_rh[:3, :]

        J_lh = pin.getFrameJacobian(self.model, self.data, self.left_hand_id, frame)
        Jdot_lh = pin.getFrameJacobianTimeVariation(self.model, self.data, self.left_hand_id, frame)
        self.J_left_hand, self.Jdot_left_hand = J_lh[:3, :], Jdot_lh[:3, :]

        # torso orientation (rotational rows 3:6)
        self.J_torso = pin.getFrameJacobian(self.model, self.data, self.torso_id, frame)[3:, :]
        self.Jdot_torso = pin.getFrameJacobianTimeVariation(self.model, self.data, self.torso_id, frame)[3:, :]

        # CoM height
        J_com = pin.jacobianCenterOfMass(self.model, self.data, q)
        self.J_com_z = J_com[2:3, :]                  # (1, nv)
        self.Jdot_com_z_dq = self.data.acom[0][2:3]   # (1,) z of J_com_dot @ dq

    def pd_control(self, q, dq, ref: WBCReference):
        # Compute current swing foot position and velocity
        swing_foot_pos = self.data.oMf[self.swing_id].translation  # world frame position (3,)
        swing_foot_vel = self.J_swing @ dq  # world frame velocity (3,)
        
        ''' Swing Foot Position and Velocity Control '''
        pos_error = ref.swing_pos - swing_foot_pos
        vel_error = ref.swing_vel - swing_foot_vel

        xdd_swing_des = self.Kp_swing * pos_error + self.Kd_swing * vel_error

        ''' Torso Orientation Control '''
        # current torso orientation as rotation matrix
        R_torso = self.data.oMf[self.torso_id].rotation
        # desired orientation: upright, yawed to the heading reference
        R_des = ref.torso_rot
        # orientation error — difference between current and desired rotation
        # standard approach is the rotation error vector
        R_error = R_des @ R_torso.T  # error rotation matrix
        # convert to axis-angle (rotation vector)
        orientation_error = pin.log3(R_error)  # shape (3,)
        # current torso angular velocity
        omega_torso = self.J_torso @ dq  # shape (3,)
        alpha_torso_des = self.Kp_torso * orientation_error + self.Kd_torso * (ref.torso_omega - omega_torso)

        '''CoM position and velocity '''
        com_pos = self.data.com[0]  # world frame CoM position (3,)
        com_vel = self.data.vcom[0]  # world frame CoM velocity (3,)

        height_error = self.com_desired_height - com_pos[2]
        com_vel_z = com_vel[2]
        zdd_com_des = self.Kp_com * height_error - self.Kd_com * com_vel_z

        ''' Swing Foot Orientation Control '''
        # swing foot orientation: level, yawed to the landing heading reference
        R_swing = self.data.oMf[self.swing_id].rotation
        R_swing_des = ref.swing_rot
        R_err_swing = R_swing_des @ R_swing.T
        swing_orient_err = pin.log3(R_err_swing)
        omega_swing = self.J_swing_rot @ dq
        alpha_swing_des = self.Kp_swing_rot * swing_orient_err + self.Kd_swing_rot * (ref.swing_omega - omega_swing)

        ''' Arm Posture Control '''
        # arm posture: joint-space PD toward q_ref
        qdd_arm_des = self.Kp_arm * (self.q_ref[self.arm_q_idx] - q[self.arm_q_idx]) - self.Kd_arm * dq[self.arm_v_idx]

        ''' Right Hand Position Control '''
        # no reference -> zero desired acceleration (the task weight is 0 in the QP)
        xdd_r_hand_des = np.zeros(3)
        if ref.right_hand_pos is not None:
            r_hand_pos = self.data.oMf[self.right_hand_id].translation  # world frame position (3,)
            r_hand_vel = self.J_right_hand @ dq                         # world frame velocity (3,)
            r_hand_vel_des = np.zeros(3) if ref.right_hand_vel is None else ref.right_hand_vel
            xdd_r_hand_des = self.Kp_hand * (ref.right_hand_pos - r_hand_pos) + self.Kd_hand * (r_hand_vel_des - r_hand_vel)

        ''' Left Hand Position Control '''
        xdd_l_hand_des = np.zeros(3)
        if ref.left_hand_pos is not None:
            l_hand_pos = self.data.oMf[self.left_hand_id].translation  # world frame position (3,)
            l_hand_vel = self.J_left_hand @ dq                         # world frame velocity (3,)
            l_hand_vel_des = np.zeros(3) if ref.left_hand_vel is None else ref.left_hand_vel
            xdd_l_hand_des = self.Kp_hand * (ref.left_hand_pos - l_hand_pos) + self.Kd_hand * (l_hand_vel_des - l_hand_vel)

        return xdd_swing_des, alpha_torso_des, zdd_com_des, alpha_swing_des, qdd_arm_des, xdd_r_hand_des, xdd_l_hand_des

    def compute_control(self, q, dq, ref: WBCReference, stance_foot):
        self.compute_jacobians(q, dq, stance_foot)
        xdd_swing_des, alpha_torso_des, zdd_com_des, alpha_swing_des, qdd_arm_des, xdd_r_hand_des, xdd_l_hand_des = self.pd_control(q, dq, ref)

        J_stance, Jdot_stance = self.J_stance, self.Jdot_stance
        J_swing, Jdot_swing = self.J_swing, self.Jdot_swing
        J_swing_rot, Jdot_swing_rot = self.J_swing_rot, self.Jdot_swing_rot
        J_torso, Jdot_torso = self.J_torso, self.Jdot_torso
        J_com_z, Jdot_com_z_dq = self.J_com_z, self.Jdot_com_z_dq
        J_r_hand, Jdot_r_hand = self.J_right_hand, self.Jdot_right_hand
        J_l_hand, Jdot_l_hand = self.J_left_hand, self.Jdot_left_hand

        # Formulate and solve QP to get q̈ and contact forces λ
        D = self.data.M
        C = self.data.C
        G = self.data.g

        nv = self.nv
        D_floating = D[:6, :] # shape: (6, nv)
        C_floating = C[:6, :] # shape: (6, nv)
        G_floating = G[:6]    # shape: (6,)

        J_stance_floating = J_stance[:, :6]  # shape: (3, 6)

        A_eq = np.zeros((9, self.n_vars))
        b_eq = np.zeros(9)

        A_eq[0:6, :nv] = D_floating
        A_eq[0:6, nv:] = -J_stance_floating.T
        A_eq[6:9, :nv] = J_stance

        b_eq[0:6] = -C_floating @ dq - G_floating
        b_eq[6:9] = -Jdot_stance @ dq

        # Known external hand forces (force ON the robot from the box, world frame).
        # Dynamics: M qdd + C dq + g = S^T tau + J_st^T lambda + J_r^T f_r + J_l^T f_l
        f_r_hand = np.zeros(3) if ref.right_hand_force is None else ref.right_hand_force
        f_l_hand = np.zeros(3) if ref.left_hand_force is None else ref.left_hand_force
        # floating-base rows: the hand forces act on the base like the stance force does
        b_eq[0:6] += J_r_hand[:, :6].T @ f_r_hand + J_l_hand[:, :6].T @ f_l_hand

        w1, w2, w3, w4, w5, w_arm = self.w_swing, self.w_torso, self.w_com, self.w_swing_rot, self.w_reg, self.w_arm
        # hand tasks are only on when they have a reference
        w_r_hand = self.w_hand if ref.right_hand_pos is not None else 0.0
        w_l_hand = self.w_hand if ref.left_hand_pos is not None else 0.0

        # P matrix
        P = np.zeros((self.n_vars, self.n_vars))
        P[:nv, :nv] += w1 * J_swing.T @ J_swing      # swing foot
        P[:nv, :nv] += w2 * J_torso.T @ J_torso      # torso orientation
        P[:nv, :nv] += w3 * J_com_z.T @ J_com_z      # CoM height
        P[:nv, :nv] += w4 * J_swing_rot.T @ J_swing_rot
        P[:nv, :nv] += w5 * np.eye(nv)               # regularization
        P[:nv, :nv] += w_arm * self.S_arm.T @ self.S_arm  # arm posture
        P[:nv, :nv] += w_r_hand * J_r_hand.T @ J_r_hand  # right hand position
        P[:nv, :nv] += w_l_hand * J_l_hand.T @ J_l_hand  # left hand position
        P[nv:, nv:] += 1e-6 * np.eye(3)  # small regularization on contact forces

        # q vector
        q_vec = np.zeros(self.n_vars)
        q_vec[:nv] += w1 * J_swing.T @ (Jdot_swing @ dq - xdd_swing_des)
        q_vec[:nv] += w2 * J_torso.T @ (Jdot_torso @ dq - alpha_torso_des)
        q_vec[:nv] += w3 * J_com_z.T @ (Jdot_com_z_dq - zdd_com_des)
        q_vec[:nv] += w4 * J_swing_rot.T @ (Jdot_swing_rot @ dq - alpha_swing_des)
        q_vec[:nv] += w5 * np.zeros(nv) # regularization — zero desired acceleration
        q_vec[:nv] += w_arm * self.S_arm.T @ (-qdd_arm_des)  # arm posture
        q_vec[:nv] += w_r_hand * J_r_hand.T @ (Jdot_r_hand @ dq - xdd_r_hand_des)  # right hand position
        q_vec[:nv] += w_l_hand * J_l_hand.T @ (Jdot_l_hand @ dq - xdd_l_hand_des)  # left hand position

        mu_pyramid = self.mu / np.sqrt(2)
        G_fric = np.zeros((5, self.n_vars))
        h_fric = np.zeros(5)
        G_fric[0, nv+2] = -1
        G_fric[1, nv]   =  1; G_fric[1, nv+2] = -mu_pyramid
        G_fric[2, nv]   = -1; G_fric[2, nv+2] = -mu_pyramid
        G_fric[3, nv+1] =  1; G_fric[3, nv+2] = -mu_pyramid
        G_fric[4, nv+1] = -1; G_fric[4, nv+2] = -mu_pyramid

        const_vec = C[6:, :] @ dq + G[6:]
        # torque-limit rows: tau = D_a qdd + C_a dq + g_a - J_st,a^T lambda - J_h,a^T f_h
        const_vec -= J_r_hand[:, 6:].T @ f_r_hand + J_l_hand[:, 6:].T @ f_l_hand
        Tau_map   = np.hstack([D[6:, :], -J_stance[:, 6:].T])
        G_tau = np.vstack([Tau_map, -Tau_map])
        h_tau = np.concatenate([
            self.tau_limits - const_vec,
            self.tau_limits + const_vec,
        ])

        G_ineq = np.vstack([G_fric, G_tau])
        h_ineq = np.concatenate([h_fric, h_tau])

        x = solve_qp(P, q_vec, A=A_eq, b=b_eq, G=G_ineq, h=h_ineq, solver="quadprog")

        if x is None:
            print("WBC QP infeasible — falling back to gravity compensation")
            return (self.data.g)[6:]


        qdd = x[:nv]
        lambda_contact = x[nv:]

        # print("lambda_contact:", lambda_contact)

        foot_acc_commanded = J_swing @ qdd + Jdot_swing @ dq

        # com = self.data.com[0]          # world-frame CoM position (3,)
        # com_xy = com[:2]
        # stance_xy = self.data.oMf[self.stance_id].translation[:2]
        # print("CoM-stance lateral offset:", com_xy - stance_xy)
        # print("lambda (contact force):", lambda_contact)

        swing_vel = J_swing @ dq
        # print("Swing Vel (Jacobian):", swing_vel)
        
        # # what the QP actually commanded for the swing foot
        # print("desired swing acc:", xdd_swing_des)
        # print("commanded swing acc:", foot_acc_commanded)
        # print()
        # ---- solve for tau ---- #
        tau_full = (D @ qdd + C @ dq + G - J_stance.T @ lambda_contact
                    - J_r_hand.T @ f_r_hand - J_l_hand.T @ f_l_hand)  # the box supplies part of the hand force
        if np.any(np.abs(tau_full[6:]) > self.tau_limits + 1e-3):
            print(f"WARNING: torque limit exceeded! max |tau|/limit = {np.max(np.abs(tau_full[6:]) / self.tau_limits):.2f}")
        tau = tau_full[6:]  # actuated joints only

        return tau

    
if __name__ == "__main__":
    wbc = WholeBodyController()

    for i, name in enumerate(wbc.model.names):
        print(f"Pin joint {i}: {name}")