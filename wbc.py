from pyexpat import model

import pinocchio as pin
import numpy as np
from qpsolvers import solve_qp
from constants import *

class WholeBodyController:

    def __init__(self, model, data, q_ref, arm_idx_q, arm_idx_dq):
        self.model = model
        self.data = data

        self.q_ref = q_ref
        self.arm_idx_q = arm_idx_q
        self.arm_idx_dq = arm_idx_dq

        self.nv = self.model.nv # total DOF
        self.n_vars = self.nv + 3 # total decision variables n-DOF plus linear contact forces (x, y, z)

        self.feet = FOOT_NAMES

        self.swing_id = None
        self.stance_id = None
        self.torso_id = self.model.getFrameId(TORSO_NAME)

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
        # Posture Regularization
        self.Kp_posture, self.Kd_posture = 100.0, 20.0

    def compute_jacobians(self, q, dq, stance_foot):
        # Get frame IDs
        self.stance_id = self.model.getFrameId(stance_foot)
        self.swing_id = self.model.getFrameId(self.feet[1] if stance_foot == self.feet[0] else self.feet[0])

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

        arm_selection_matrix = np.zeros((len(self.arm_idx_dq), self.model.nv))
        arm_selection_matrix[np.arange(len(self.arm_idx_dq)), self.arm_idx_dq] = 1.0 # Selection matrix for arm joints
       
        return  J_stance, Jdot_stance, J_swing, Jdot_swing, J_swing_rot, Jdot_swing_rot, J_torso, Jdot_torso, J_com_z, Jdot_com_z_dq, arm_selection_matrix
    
    def pd_control(self, q, dq, target_pos, target_vel, J_swing, J_torso, J_swing_rot):
        # Compute current swing foot position and velocity
        swing_foot_pos = self.data.oMf[self.swing_id].translation  # world frame position (3,)
        swing_foot_vel = J_swing @ dq  # world frame velocity (3,)
        # Desired accelerations (PD control)
        pos_error = target_pos - swing_foot_pos
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


        q_ref_arms = self.q_ref[self.arm_idx_q]        # reference angles, from your keyframe
        q_arms = q[self.arm_idx_q]
        dq_arms = dq[self.arm_idx_dq]

        qdd_posture_des = self.Kp_posture * (q_ref_arms - q_arms) - self.Kd_posture * dq_arms

        return xdd_swing_des, alpha_torso_des, zdd_com_des, alpha_swing_des, qdd_posture_des
    
    def compute_control(self, q, dq, target_pos, target_vel, stance_foot):
        J_stance, Jdot_stance, J_swing, Jdot_swing, J_swing_rot, Jdot_swing_rot, J_torso, Jdot_torso, J_com_z, Jdot_com_z_dq, arm_selection_matrix = self.compute_jacobians(q, dq, stance_foot)
        xdd_swing_des, alpha_torso_des, zdd_com_des, alpha_swing_des, qdd_posture_des = self.pd_control(q, dq, target_pos, target_vel, J_swing, J_torso, J_swing_rot)

        # Formulate and solve QP to get q̈ and contact forces λ
        D = self.data.M
        C = self.data.C
        G = self.data.g

        D_floating = D[:6, :] # shape: (6, 16)
        C_floating = C[:6, :] # shape: (6, 16)
        G_floating = G[:6]    # shape: (6,)

        J_stance_floating = J_stance[:, :6]  # shape: (3, 6)

        A_eq = np.zeros((9, self.n_vars))
        b_eq = np.zeros(9)

        A_eq[0:6, 0:self.nv] = D_floating
        A_eq[0:6, self.nv:self.n_vars] = -J_stance_floating.T
        A_eq[6:9, 0:self.nv] = J_stance

        b_eq[0:6] = -C_floating @ dq - G_floating
        b_eq[6:9] = -Jdot_stance @ dq

        w_swing_pos, w_torso_orientation, w_com_height, w_swing_orientation, w_posture, w_reg = 200.0, 8.0, 1.0, 1.0, 1.0, 0.01

        # P matrix
        P = np.zeros((self.n_vars, self.n_vars))
        P[0:self.nv, 0:self.nv] += w_swing_pos * J_swing.T @ J_swing      # Swing Foot Position
        P[0:self.nv, 0:self.nv] += w_torso_orientation * J_torso.T @ J_torso      # Torso Orientation
        P[0:self.nv, 0:self.nv] += w_com_height * J_com_z.T @ J_com_z      # CoM height
        P[0:self.nv, 0:self.nv] += w_swing_orientation * J_swing_rot.T @ J_swing_rot # Swing Foot Orientation
        P[0:self.nv, 0:self.nv] += w_posture * arm_selection_matrix.T @ arm_selection_matrix # Posture Regularization
        P[0:self.nv, 0:self.nv] += w_reg * np.eye(self.nv)               # Regularization
        P[self.nv:self.n_vars, self.nv:self.n_vars] += 1e-6 * np.eye(3)  # small regularization on contact forces

        # q vector
        q_vec = np.zeros(self.n_vars)
        q_vec[0:self.nv] += w_swing_pos * J_swing.T @ (Jdot_swing @ dq - xdd_swing_des)
        q_vec[0:self.nv] += w_torso_orientation * J_torso.T @ (Jdot_torso @ dq - alpha_torso_des)
        q_vec[0:self.nv] += w_com_height * J_com_z.T @ (Jdot_com_z_dq - zdd_com_des)
        q_vec[0:self.nv] += w_swing_orientation * J_swing_rot.T @ (Jdot_swing_rot @ dq - alpha_swing_des)
        q_vec[0:self.nv] += w_posture * arm_selection_matrix.T @ (-qdd_posture_des)
        q_vec[0:self.nv] += w_reg * np.zeros(self.nv) # regularization — zero desired acceleration

        mu_pyramid = self.mu / np.sqrt(2)
        G_fric = np.zeros((5, self.n_vars))
        h_fric = np.zeros(5)
        G_fric[0, self.n_vars-1] = -1
        G_fric[1, self.nv]   =  1; G_fric[1, self.n_vars-1] = -mu_pyramid
        G_fric[2, self.nv]   = -1; G_fric[2, self.n_vars-1] = -mu_pyramid
        G_fric[3, self.nv+1] =  1; G_fric[3, self.n_vars-1] = -mu_pyramid
        G_fric[4, self.nv+1] = -1; G_fric[4, self.n_vars-1] = -mu_pyramid

        const_vec = C[6:, :] @ dq + G[6:]
        Tau_map   = np.hstack([D[6:, :], -J_stance[:, 6:].T])
        G_tau = np.vstack([Tau_map, -Tau_map])
        h_tau = np.concatenate([
            ACTUATOR_LIMITS - const_vec,
            ACTUATOR_LIMITS + const_vec,
        ])

        # Joint Limits 
        joint_limits_lower = self.model.lowerPositionLimit[7:]   # skip floating base's 7 q-entries
        joint_limits_upper = self.model.upperPositionLimit[7:]

        dt_lookahead = 0.1  # tune — bigger gives smoother deceleration approaching the limit

        n_joints = self.nv - 6  # actuated joints only
        G_jointlim = np.zeros((2 * n_joints, self.n_vars))
        h_jointlim = np.zeros(2 * n_joints)

        for i in range(n_joints):
            qi = q[7 + i]      # position, offset for floating-base q layout
            dqi = dq[6 + i]     # velocity, offset for floating-base v layout
            q_min, q_max = joint_limits_lower[i], joint_limits_upper[i]

            # upper bound
            G_jointlim[2*i, 6+i] = 1
            h_jointlim[2*i] = (2/dt_lookahead**2) * (q_max - qi - dt_lookahead*dqi)

            # lower bound
            G_jointlim[2*i+1, 6+i] = -1
            h_jointlim[2*i+1] = -(2/dt_lookahead**2) * (q_min - qi - dt_lookahead*dqi)

        G_ineq = np.vstack([G_fric, G_tau, G_jointlim])
        h_ineq = np.concatenate([h_fric, h_tau, h_jointlim])

        x = solve_qp(P, q_vec, A=A_eq, b=b_eq, G=G_ineq, h=h_ineq, solver="quadprog")

        if x is None:
            print("WBC QP infeasible — falling back to gravity compensation")
            return (self.data.g)[6:]


        qdd = x[:self.nv]
        lambda_contact = x[self.nv:self.n_vars]

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

        # max_torque = np.abs(tau_full[6:]).max()
        # if max_torque > ACTUATOR_LIMIT + 1e-3:
        #     print(f"WARNING: torque limit exceeded! max torque = {max_torque:.2f} Nm")

        tau = tau_full[6:]  # actuated joints only

        return tau

    
if __name__ == "__main__":
    wbc = WholeBodyController()

    for i, name in enumerate(wbc.model.names):
        print(f"Pin joint {i}: {name}")