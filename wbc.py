from pyexpat import model

import pinocchio as pin
import numpy as np
from qpsolvers import solve_qp
from constants import *

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

    def compute_jacobians(self, q, dq, stance_foot):
        # Get frame IDs
        self.stance_id = self.model.getFrameId(stance_foot, pin.FrameType.BODY)
        self.swing_id = self.model.getFrameId("left_foot" if stance_foot == "right_foot" else "right_foot", pin.FrameType.BODY)

        pin.computeJointJacobians(self.model, self.data, q)
        pin.computeJointJacobiansTimeVariation(self.model, self.data, q, dq)

        # stance foot
        J_stance = pin.getFrameJacobian(self.model, self.data, self.stance_id, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        Jdot_stance = pin.getFrameJacobianTimeVariation(self.model, self.data, self.stance_id, pin.LOCAL_WORLD_ALIGNED)[:3, :]

        # swing foot
        J_swing = pin.getFrameJacobian(self.model, self.data, self.swing_id, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        Jdot_swing = pin.getFrameJacobianTimeVariation(self.model, self.data, self.swing_id, pin.LOCAL_WORLD_ALIGNED)[:3, :]

        # swing foot orientation (rotational rows 3:6)
        J_swing_rot = pin.getFrameJacobian(self.model, self.data, self.swing_id, pin.LOCAL_WORLD_ALIGNED)[3:, :]
        Jdot_swing_rot = pin.getFrameJacobianTimeVariation(self.model, self.data, self.swing_id, pin.LOCAL_WORLD_ALIGNED)[3:, :]

        # torso orientation (rotational rows 3:6)
        J_torso = pin.getFrameJacobian(self.model, self.data, self.torso_id, pin.LOCAL_WORLD_ALIGNED)[3:, :]
        Jdot_torso = pin.getFrameJacobianTimeVariation(self.model, self.data, self.torso_id, pin.LOCAL_WORLD_ALIGNED)[3:, :]

        # compute CoM Jacobian
        J_com = pin.jacobianCenterOfMass(self.model, self.data, q)
        J_com_z = J_com[2:3, :]  # shape (1, nv)
        Jdot_com_dq = self.data.acom[0]  # shape (3,) — this is J_com_dot @ dq
        Jdot_com_z_dq = Jdot_com_dq[2:3]  # z component only
       
        return  J_stance, Jdot_stance, J_swing, Jdot_swing, J_swing_rot, Jdot_swing_rot, J_torso, Jdot_torso, J_com_z, Jdot_com_z_dq
    
    def pd_control(self, q, dq, target_pos, target_vel, J_swing, J_torso, J_swing_rot):
        # Compute current swing foot position and velocity
        swing_foot_pos = self.data.oMf[self.swing_id].translation  # world frame position (3,)
        swing_foot_vel = J_swing @ dq  # world frame velocity (3,)
        # Desired accelerations (PD control)
        # print("target_pos:", target_pos)
        # print("swing_foot_pos:", swing_foot_pos)
        pos_error = target_pos - swing_foot_pos
        # print("pos_error:", pos_error)
        vel_error = target_vel - swing_foot_vel  # desired velocity is zero at target
        xdd_swing_des = self.Kp_swing * pos_error + self.Kd_swing * vel_error

        # current torso orientation as rotation matrix
        R_torso = self.data.oMf[self.torso_id].rotation
        # desired orientation is identity (upright)
        R_des = np.eye(3)
        # orientation error — difference between current and desired rotation
        # standard approach is the rotation error vector
        R_error = R_des @ R_torso.T  # error rotation matrix
        # convert to axis-angle (rotation vector)
        orientation_error = pin.log3(R_error)  # shape (3,)
        # current torso angular velocity
        omega_torso = J_torso @ dq  # shape (3,)
        alpha_torso_des = self.Kp_torso * orientation_error - self.Kd_torso * omega_torso

        # Current CoM position and velocity
        com_pos = self.data.com[0]  # world frame CoM position (3,)
        com_vel = self.data.vcom[0]  # world frame CoM velocity (3,)
        # Desired accelerations (PD control)
        height_error = self.com_desired_height - com_pos[2]
        com_vel_z = com_vel[2]
        zdd_com_des = self.Kp_com * height_error - self.Kd_com * com_vel_z

        # swing foot orientation: keep foot level (identity target)
        R_swing = self.data.oMf[self.swing_id].rotation
        R_swing_des = np.eye(3)               # level foot
        R_err_swing = R_swing_des @ R_swing.T
        swing_orient_err = pin.log3(R_err_swing)
        omega_swing = J_swing_rot @ dq
        alpha_swing_des = self.Kp_swing_rot * swing_orient_err - self.Kd_swing_rot * omega_swing

        # arm posture: joint-space PD toward q_ref
        qdd_arm_des = self.Kp_arm * (self.q_ref[self.arm_q_idx] - q[self.arm_q_idx]) - self.Kd_arm * dq[self.arm_v_idx]

        return xdd_swing_des, alpha_torso_des, zdd_com_des, alpha_swing_des, qdd_arm_des
    
    def compute_control(self, q, dq, target_pos, target_vel, stance_foot):
        J_stance, Jdot_stance, J_swing, Jdot_swing, J_swing_rot, Jdot_swing_rot, J_torso, Jdot_torso, J_com_z, Jdot_com_z_dq = self.compute_jacobians(q, dq, stance_foot)
        xdd_swing_des, alpha_torso_des, zdd_com_des, alpha_swing_des, qdd_arm_des = self.pd_control(q, dq, target_pos, target_vel, J_swing, J_torso, J_swing_rot)

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

        w1, w2, w3, w4, w5, w_arm = 200.0, 8.0, 1.0, 1.0, 0.01, 1.0

        # P matrix
        P = np.zeros((self.n_vars, self.n_vars))
        P[:nv, :nv] += w1 * J_swing.T @ J_swing      # swing foot
        P[:nv, :nv] += w2 * J_torso.T @ J_torso      # torso orientation
        P[:nv, :nv] += w3 * J_com_z.T @ J_com_z      # CoM height
        P[:nv, :nv] += w4 * J_swing_rot.T @ J_swing_rot
        P[:nv, :nv] += w5 * np.eye(nv)               # regularization
        P[:nv, :nv] += w_arm * self.S_arm.T @ self.S_arm  # arm posture
        P[nv:, nv:] += 1e-6 * np.eye(3)  # small regularization on contact forces

        # q vector
        q_vec = np.zeros(self.n_vars)
        q_vec[:nv] += w1 * J_swing.T @ (Jdot_swing @ dq - xdd_swing_des)
        q_vec[:nv] += w2 * J_torso.T @ (Jdot_torso @ dq - alpha_torso_des)
        q_vec[:nv] += w3 * J_com_z.T @ (Jdot_com_z_dq - zdd_com_des)
        q_vec[:nv] += w4 * J_swing_rot.T @ (Jdot_swing_rot @ dq - alpha_swing_des)
        q_vec[:nv] += w5 * np.zeros(nv) # regularization — zero desired acceleration
        q_vec[:nv] += w_arm * self.S_arm.T @ (-qdd_arm_des)  # arm posture

        mu_pyramid = self.mu / np.sqrt(2)
        G_fric = np.zeros((5, self.n_vars))
        h_fric = np.zeros(5)
        G_fric[0, nv+2] = -1
        G_fric[1, nv]   =  1; G_fric[1, nv+2] = -mu_pyramid
        G_fric[2, nv]   = -1; G_fric[2, nv+2] = -mu_pyramid
        G_fric[3, nv+1] =  1; G_fric[3, nv+2] = -mu_pyramid
        G_fric[4, nv+1] = -1; G_fric[4, nv+2] = -mu_pyramid

        const_vec = C[6:, :] @ dq + G[6:]
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
        tau_full = D @ qdd + C @ dq + G - J_stance.T @ lambda_contact
        if np.any(np.abs(tau_full[6:]) > self.tau_limits + 1e-3):
            print(f"WARNING: torque limit exceeded! max |tau|/limit = {np.max(np.abs(tau_full[6:]) / self.tau_limits):.2f}")
        tau = tau_full[6:]  # actuated joints only

        return tau

    
if __name__ == "__main__":
    wbc = WholeBodyController()

    for i, name in enumerate(wbc.model.names):
        print(f"Pin joint {i}: {name}")