from pyexpat import model

import pinocchio as pin
import numpy as np
from qpsolvers import solve_qp

class WholeBodyController:

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.feet = ["right_foot", "left_foot"]

        self.swing_id = None
        self.stance_id = None
        self.torso_id = self.model.getFrameId("torso")

        self.com_desired_height = 1.45

        # PD Controller gains
        self.Kp = 100.0 * 5
        self.Kd = 20.0 * 5

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

        # torso orientation (rotational rows 3:6)
        J_torso = pin.getFrameJacobian(self.model, self.data, self.torso_id, pin.LOCAL_WORLD_ALIGNED)[3:, :]
        Jdot_torso = pin.getFrameJacobianTimeVariation(self.model, self.data, self.torso_id, pin.LOCAL_WORLD_ALIGNED)[3:, :]

        # compute CoM Jacobian
        J_com = pin.jacobianCenterOfMass(self.model, self.data, q)
        J_com_z = J_com[2:3, :]  # shape (1, 16)
        Jdot_com_dq = self.data.acom[0]  # shape (3,) — this is J_com_dot @ dq
        Jdot_com_z_dq = Jdot_com_dq[2:3]  # z component only
       
        return  J_stance, Jdot_stance, J_swing, Jdot_swing, J_torso, Jdot_torso, J_com_z, Jdot_com_z_dq
    
    def pd_control(self, q, dq, target_pos, J_swing, J_torso):
        # Compute current swing foot position and velocity
        swing_foot_pos = self.data.oMf[self.swing_id].translation  # world frame position (3,)
        swing_foot_vel = J_swing @ dq  # world frame velocity (3,)
        # Desired accelerations (PD control)
        # print("target_pos:", target_pos)
        # print("swing_foot_pos:", swing_foot_pos)
        pos_error = target_pos - swing_foot_pos
        # print("pos_error:", pos_error)
        vel_error = -swing_foot_vel  # desired velocity is zero at target
        xdd_swing_des = self.Kp * pos_error + self.Kd * vel_error

        # print("Desired xdd: ", xdd_swing_des)

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
        alpha_torso_des = self.Kp * orientation_error - self.Kd * omega_torso

        # Current CoM position and velocity
        com_pos = self.data.com[0]  # world frame CoM position (3,)
        com_vel = self.data.vcom[0]  # world frame CoM velocity (3,)
        # Desired accelerations (PD control)
        height_error = self.com_desired_height - com_pos[2]
        com_vel_z = com_vel[2]
        zdd_com_des = self.Kp * height_error - self.Kd * com_vel_z

        return xdd_swing_des, alpha_torso_des, zdd_com_des
    
    def compute_control(self, q, dq, target_pos, stance_foot):
        J_stance, Jdot_stance, J_swing, Jdot_swing, J_torso, Jdot_torso, J_com_z, Jdot_com_z_dq = self.compute_jacobians(q, dq, stance_foot)
        xdd_swing_des, alpha_torso_des, zdd_com_des = self.pd_control(q, dq, target_pos, J_swing, J_torso)

        # Formulate and solve QP to get q̈ and contact forces λ
        D = self.data.M
        C = self.data.C
        G = self.data.g

        D_floating = D[:6, :] # shape: (6, 16)
        C_floating = C[:6, :] # shape: (6, 16)
        G_floating = G[:6]    # shape: (6,)

        J_stance_floating = J_stance[:, :6]  # shape: (3, 6)

        A_eq = np.zeros((9, 19))
        b_eq = np.zeros(9)

        A_eq[0:6, 0:16] = D_floating
        A_eq[0:6, 16:19] = -J_stance_floating.T
        A_eq[6:9, 0:16] = J_stance

        b_eq[0:6] = -C_floating @ dq - G_floating
        b_eq[6:9] = -Jdot_stance @ dq

        w1, w2, w3, w4 = 1.0, 1.0, 1.0, 0.01

        # P matrix
        P = np.zeros((19, 19))
        P[0:16, 0:16] += w1 * J_swing.T @ J_swing      # swing foot
        P[0:16, 0:16] += w2 * J_torso.T @ J_torso      # torso orientation
        P[0:16, 0:16] += w3 * J_com_z.T @ J_com_z      # CoM height
        P[0:16, 0:16] += w4 * np.eye(16)               # regularization
        P[16:19, 16:19] += 1e-6 * np.eye(3)  # small regularization on contact forces

        # q vector
        q_vec = np.zeros(19)
        q_vec[0:16] += w1 * J_swing.T @ (Jdot_swing @ dq - xdd_swing_des)
        q_vec[0:16] += w2 * J_torso.T @ (Jdot_torso @ dq - alpha_torso_des)
        q_vec[0:16] += w3 * J_com_z.T @ (Jdot_com_z_dq - zdd_com_des)
        q_vec[0:16] += w4 * np.zeros(16) # regularization — zero desired acceleration

        # inequality constraint: lambda_z >= 0
        G_ineq = np.zeros((1, 19))
        G_ineq[0, 18] = -1  # -lambda_z <= 0  →  lambda_z >= 0
        h_ineq = np.zeros(1)

        x = solve_qp(P, q_vec, A=A_eq, b=b_eq, G=G_ineq, h=h_ineq, solver="quadprog")

        qdd = x[:16]
        lambda_contact = x[16:]

        # print("lambda_contact:", lambda_contact)

        foot_acc_commanded = J_swing @ qdd + Jdot_swing @ dq
        
        # # what the QP actually commanded for the swing foot
        # print("desired swing acc:", xdd_swing_des)
        # print("commanded swing acc:", foot_acc_commanded)
        # print("stance constraint foot:", self.stance_id, " swing foot:", self.swing_id)

        # ---- solve for tau ---- #
        tau_full = D @ qdd + C @ dq + G - J_stance.T @ lambda_contact
        tau = tau_full[6:]  # actuated joints only

        return tau

if __name__ == "__main__":
    wbc = WholeBodyController()

    for i, name in enumerate(wbc.model.names):
        print(f"Pin joint {i}: {name}")