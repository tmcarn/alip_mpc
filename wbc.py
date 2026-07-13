from pyexpat import model

import pinocchio as pin
import numpy as np
from qpsolvers import solve_qp
from constants import *

class WholeBodyController:

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.feet = ["right_foot", "left_foot"]

        self.swing_id = None
        self.stance_id = None
        self.torso_id = self.model.getFrameId("torso")

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

    def compute_jacobians(self, q, dq, stance_foot):
        # Get frame IDs
        self.stance_id = self.model.getFrameId(stance_foot)
        self.swing_id = self.model.getFrameId("left_foot" if stance_foot == "right_foot" else "right_foot")

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
        J_com_z = J_com[2:3, :]  # shape (1, 16)
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

        return xdd_swing_des, alpha_torso_des, zdd_com_des, alpha_swing_des
    
    # def compute_control(self, q, dq, target_pos, target_vel, stance_foot):
    #     J_stance, Jdot_stance, J_swing, Jdot_swing, J_swing_rot, Jdot_swing_rot, J_torso, Jdot_torso, J_com_z, Jdot_com_z_dq = self.compute_jacobians(q, dq, stance_foot)
    #     xdd_swing_des, alpha_torso_des, zdd_com_des, alpha_swing_des = self.pd_control(q, dq, target_pos, target_vel, J_swing, J_torso, J_swing_rot)

    #     # Formulate and solve QP to get q̈ and contact forces λ
    #     D = self.data.M
    #     C = self.data.C
    #     G = self.data.g

    #     D_floating = D[:6, :] # shape: (6, 16)
    #     C_floating = C[:6, :] # shape: (6, 16)
    #     G_floating = G[:6]    # shape: (6,)

    #     J_stance_floating = J_stance[:, :6]  # shape: (3, 6)

    #     A_eq = np.zeros((9, 19))
    #     b_eq = np.zeros(9)

    #     A_eq[0:6, 0:16] = D_floating
    #     A_eq[0:6, 16:19] = -J_stance_floating.T
    #     A_eq[6:9, 0:16] = J_stance

    #     b_eq[0:6] = -C_floating @ dq - G_floating
    #     b_eq[6:9] = -Jdot_stance @ dq

    #     w1, w2, w3, w4, w5 = 200.0, 8.0, 1.0, 1.0, 0.01

    #     # P matrix
    #     P = np.zeros((19, 19))
    #     P[0:16, 0:16] += w1 * J_swing.T @ J_swing      # swing foot
    #     P[0:16, 0:16] += w2 * J_torso.T @ J_torso      # torso orientation
    #     P[0:16, 0:16] += w3 * J_com_z.T @ J_com_z      # CoM height
    #     P[0:16, 0:16] += w4 * J_swing_rot.T @ J_swing_rot
    #     P[0:16, 0:16] += w5 * np.eye(16)               # regularization
    #     P[16:19, 16:19] += 1e-6 * np.eye(3)  # small regularization on contact forces

    #     # q vector
    #     q_vec = np.zeros(19)
    #     q_vec[0:16] += w1 * J_swing.T @ (Jdot_swing @ dq - xdd_swing_des)
    #     q_vec[0:16] += w2 * J_torso.T @ (Jdot_torso @ dq - alpha_torso_des)
    #     q_vec[0:16] += w3 * J_com_z.T @ (Jdot_com_z_dq - zdd_com_des)
    #     q_vec[0:16] += w4 * J_swing_rot.T @ (Jdot_swing_rot @ dq - alpha_swing_des)
    #     q_vec[0:16] += w5 * np.zeros(16) # regularization — zero desired acceleration

    #     mu_pyramid = self.mu / np.sqrt(2)
    #     G_fric = np.zeros((5, 19))
    #     h_fric = np.zeros(5)
    #     G_fric[0, 18] = -1
    #     G_fric[1, 16] =  1; G_fric[1, 18] = -mu_pyramid
    #     G_fric[2, 16] = -1; G_fric[2, 18] = -mu_pyramid
    #     G_fric[3, 17] =  1; G_fric[3, 18] = -mu_pyramid
    #     G_fric[4, 17] = -1; G_fric[4, 18] = -mu_pyramid

    #     const_vec = C[6:, :] @ dq + G[6:]
    #     Tau_map   = np.hstack([D[6:, :], -J_stance[:, 6:].T])
    #     G_tau = np.vstack([Tau_map, -Tau_map])
    #     h_tau = np.concatenate([
    #         ACTUATOR_LIMIT - const_vec,
    #         ACTUATOR_LIMIT + const_vec,
    #     ])

    #     G_ineq = np.vstack([G_fric, G_tau])
    #     h_ineq = np.concatenate([h_fric, h_tau])

    #     x = solve_qp(P, q_vec, A=A_eq, b=b_eq, G=G_ineq, h=h_ineq, solver="quadprog")

    #     if x is None:
    #         print("WBC QP infeasible — falling back to gravity compensation")
    #         return (self.data.g)[6:]


    #     qdd = x[:16]
    #     lambda_contact = x[16:]

    #     # print("lambda_contact:", lambda_contact)

    #     foot_acc_commanded = J_swing @ qdd + Jdot_swing @ dq

    #     # com = self.data.com[0]          # world-frame CoM position (3,)
    #     # com_xy = com[:2]
    #     # stance_xy = self.data.oMf[self.stance_id].translation[:2]
    #     # print("CoM-stance lateral offset:", com_xy - stance_xy)
    #     # print("lambda (contact force):", lambda_contact)

    #     swing_vel = J_swing @ dq
    #     # print("Swing Vel (Jacobian):", swing_vel)
        
    #     # # what the QP actually commanded for the swing foot
    #     # print("desired swing acc:", xdd_swing_des)
    #     # print("commanded swing acc:", foot_acc_commanded)
    #     # print()
    #     # ---- solve for tau ---- #
    #     tau_full = D @ qdd + C @ dq + G - J_stance.T @ lambda_contact
    #     max_torque = np.abs(tau_full[6:]).max()
    #     if max_torque > ACTUATOR_LIMIT + 1e-3:
    #         print(f"WARNING: torque limit exceeded! max torque = {max_torque:.2f} Nm")
    #     tau = tau_full[6:]  # actuated joints only

    #     return tau

    def compute_control(self, q, dq, target_pos, target_vel, stance_foot):
        J_stance, Jdot_stance, J_swing, Jdot_swing, J_swing_rot, Jdot_swing_rot, J_torso, Jdot_torso, J_com_z, Jdot_com_z_dq = self.compute_jacobians(q, dq, stance_foot)
        xdd_swing_des, alpha_torso_des, zdd_com_des, alpha_swing_des = self.pd_control(q, dq, target_pos, target_vel, J_swing, J_torso, J_swing_rot)

        D = self.data.M
        C = self.data.C
        G = self.data.g

        D_floating = D[:6, :]
        C_floating = C[:6, :]
        G_floating = G[:6]

        J_stance_floating = J_stance[:, :6]

        N_X = 19       # qdd(16) + lambda(3), same as before
        N_SLACK = 12   # s_fric_x, s_fric_y, s_tau(10)
        N_TOTAL = N_X + N_SLACK

        # --- equality constraints (unchanged, just padded for the new slack columns) ---
        A_eq = np.zeros((9, N_TOTAL))
        b_eq = np.zeros(9)

        A_eq[0:6, 0:16] = D_floating
        A_eq[0:6, 16:19] = -J_stance_floating.T
        A_eq[6:9, 0:16] = J_stance

        b_eq[0:6] = -C_floating @ dq - G_floating
        b_eq[6:9] = -Jdot_stance @ dq

        w1, w2, w3, w4, w5 = 200.0, 8.0, 1.0, 1.0, 0.01   # <- double check this w1 matches what you intend; was 230.0 in your last paste

        # --- P matrix ---
        P = np.zeros((N_TOTAL, N_TOTAL))
        P[0:16, 0:16] += w1 * J_swing.T @ J_swing
        P[0:16, 0:16] += w2 * J_torso.T @ J_torso
        P[0:16, 0:16] += w3 * J_com_z.T @ J_com_z
        P[0:16, 0:16] += w4 * J_swing_rot.T @ J_swing_rot
        P[0:16, 0:16] += w5 * np.eye(16)
        P[16:19, 16:19] += 1e-6 * np.eye(3)
        P[N_X:, N_X:] += 1e-6 * np.eye(N_SLACK)   # tiny reg on slacks, purely for conditioning

        # --- q vector ---
        q_vec = np.zeros(N_TOTAL)
        q_vec[0:16] += w1 * J_swing.T @ (Jdot_swing @ dq - xdd_swing_des)
        q_vec[0:16] += w2 * J_torso.T @ (Jdot_torso @ dq - alpha_torso_des)
        q_vec[0:16] += w3 * J_com_z.T @ (Jdot_com_z_dq - zdd_com_des)
        q_vec[0:16] += w4 * J_swing_rot.T @ (Jdot_swing_rot @ dq - alpha_swing_des)
        q_vec[0:16] += w5 * np.zeros(16)

        w_slack = 1e5   # heavy: only use slack when truly necessary
        q_vec[N_X:] = w_slack

        # --- friction cone (soft on the four tangential bounds; lambda_z>=0 stays hard) ---
        mu_pyramid = self.mu / np.sqrt(2)
        G_fric = np.zeros((5, N_TOTAL))
        h_fric = np.zeros(5)

        G_fric[0, 18] = -1   # -lambda_z <= 0  (hard, never relaxed)

        G_fric[1, 16] =  1; G_fric[1, 18] = -mu_pyramid; G_fric[1, 19] = -1
        G_fric[2, 16] = -1; G_fric[2, 18] = -mu_pyramid; G_fric[2, 19] = -1
        G_fric[3, 17] =  1; G_fric[3, 18] = -mu_pyramid; G_fric[3, 20] = -1
        G_fric[4, 17] = -1; G_fric[4, 18] = -mu_pyramid; G_fric[4, 20] = -1

        # --- torque limits (soft, one slack per actuated joint) ---
        const_vec = C[6:, :] @ dq + G[6:]
        Tau_map   = np.hstack([D[6:, :], -J_stance[:, 6:].T])   # (10, 19)

        G_tau_up = np.zeros((10, N_TOTAL))
        G_tau_lo = np.zeros((10, N_TOTAL))
        for i in range(10):
            G_tau_up[i, 0:19] = Tau_map[i, :]
            G_tau_up[i, 21 + i] = -1
            G_tau_lo[i, 0:19] = -Tau_map[i, :]
            G_tau_lo[i, 21 + i] = -1
        h_tau_up = ACTUATOR_LIMIT - const_vec
        h_tau_lo = ACTUATOR_LIMIT + const_vec

        # --- slack non-negativity ---
        G_slack = np.zeros((N_SLACK, N_TOTAL))
        for j in range(N_SLACK):
            G_slack[j, N_X + j] = -1
        h_slack = np.zeros(N_SLACK)

        G_ineq = np.vstack([G_fric, G_tau_up, G_tau_lo, G_slack])
        h_ineq = np.concatenate([h_fric, h_tau_up, h_tau_lo, h_slack])

        x = solve_qp(P, q_vec, A=A_eq, b=b_eq, G=G_ineq, h=h_ineq, solver="quadprog")

        if x is None:
            # should be rare now -- equality constraints alone are essentially always
            # solvable; this is the genuinely pathological case (e.g. singular D)
            print("WBC QP infeasible even with slack -- falling back to gravity compensation")
            return (self.data.g)[6:]

        qdd = x[:16]
        lambda_contact = x[16:19]
        slack = x[19:]

        if np.max(slack) > 1e-3:
            print(f"WBC constraint relaxed by {np.max(slack):.3f} -- task demand exceeds physical limit")

        tau_full = D @ qdd + C @ dq + G - J_stance.T @ lambda_contact
        tau = tau_full[6:]

        return tau
    
if __name__ == "__main__":
    wbc = WholeBodyController()

    for i, name in enumerate(wbc.model.names):
        print(f"Pin joint {i}: {name}")