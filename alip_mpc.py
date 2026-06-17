import numpy as np
import scipy.linalg
from qpsolvers import solve_qp
import matplotlib.pyplot as plt
import pinocchio as pin
from constants import *


def make_A(m, g, z_H):
    A = np.zeros((4, 4))
    A[0, 3] = 1/(m*z_H)
    A[1, 2] = -1/(m*z_H)
    A[2, 1] = -m*g
    A[3, 0] = m*g
    return A

def make_B():
    B = np.zeros((4, 2))
    B[0, 0] = -1
    B[1, 1] = -1
    return B

def make_phi(Ad, H):
    phi = np.zeros((4*H, 4))
    for i in range(H):
        phi[4*i:4*(i+1), :] = np.linalg.matrix_power(Ad, i+1)
    return phi

def make_gamma(Ad, B, H):
    gamma = np.zeros((4*H, 2*H))
    for i in range(H):
        for j in range(i+1):
            gamma[4*i:4*(i+1), 2*j:2*(j+1)] = np.linalg.matrix_power(Ad, i-j+1) @ B
    return gamma

def make_Q(H, q_L):
    Q_block = np.diag([0, 0, q_L, q_L])
    return np.kron(np.eye(H), Q_block)

class ALIP_MPC:
    def __init__(self, T_s, H=10, dt_control=0.02):
        model, _, _ = pin.buildModelsFromMJCF("xml_files/biped_3d_5dof_leg.xml")
        
        data = model.createData()

        # Determine CoM for the Model
        q0 = pin.neutral(model)
        q0[2] = 1.6  # set floating base to standing height from your MJCF
        pin.centerOfMass(model, data, q0)

        self.m = data.mass[0]
        self.g = 9.81
        self.z_H = data.com[0][2]  # CoM height above ground
        self.W = STEP_WIDTH

        print(f"ALIP parameters: z_H={self.z_H:.2f} m")
        
        self.H = H
        self.T_s = T_s
        self.dt_control = dt_control

        self.A = make_A(self.m, self.g, self.z_H)
        self.B = make_B()
        self.Ad = scipy.linalg.expm(self.A * T_s)
        self.Ad_small = scipy.linalg.expm(self.A * dt_control)
        self.phi = make_phi(self.Ad, H)
        self.gamma = make_gamma(self.Ad, self.B, H)
        self.Q = make_Q(H, q_L=1.0)
        

    def get_bounds(self, stance_foot, u_lim=0.8, step_width_min=0.2): # 0.2
        bounds = []
        # sign for step 0: swing foot is opposite the stance foot
        swing_sign = +1 if stance_foot == "right_foot" else -1
        for i in range(self.H):
            bounds.append((-u_lim, u_lim))  # x
            sign = swing_sign * ((-1) ** i)  # alternates each step in horizon
            if sign > 0:
                bounds.append((step_width_min, u_lim))
            else:
                bounds.append((-u_lim, -step_width_min))
        return bounds
    
    # def make_X_ref(self, cmd_vel, m, z_H, H):
    #     X_ref = np.zeros((4*H,))
    #     l = np.sqrt(self.g / z_H)
    #     for i in range(H):
    #         X_ref[4*i + 0] = (1 / m * z_H * l) * 
    #         X_ref[4*i + 1] = 0
    #         X_ref[4*i + 2] = -m * z_H * cmd_vel[1]
    #         X_ref[4*i + 3] = m * z_H * cmd_vel[0]

    #     print("ALIP DES:", X_ref[0:4])

    #     return X_ref

    def lateral_orbit_closed_form(self, sigma):
        '''
        Gibson et al. eq (17), lateral entries only (yc, Lx), for the
        zero-lateral-velocity periodic orbit. sigma = +1 / -1 selects stance side.
        Returns (yc_des, Lx_des) for a single step.
        '''
        ell = np.sqrt(self.g / self.z_H)
        th = np.tanh(ell * self.T_s / 2.0)
        yc_des = -0.5 * sigma * self.W
        Lx_des = 0.5 * sigma * self.m * self.z_H * ell * self.W * th
        return yc_des, Lx_des


    # def lateral_orbit_expm(self, m, z_H, T_s, W, g=9.81):
    #     '''
    #     Independent derivation: solve (Phi + I) z0 = [w, 0] for the touchdown
    #     lateral state, w = +W (current-stance convention). Returns (yc, Lx)
    #     for sigma = +1; the closed form should match up to the sigma sign.
    #     '''
    #     A_lat = np.array([[0.0,   -1.0 / (m * z_H)],
    #                     [-m * g, 0.0           ]])
    #     Phi = scipy.linalg.expm(A_lat * T_s)
    #     z0 = np.linalg.solve(Phi + np.eye(2), np.array([W, 0.0]))
    #     return z0[0], z0[1]   # yc, Lx


    def make_X_ref(self, cmd_vel, stance_foot):
        '''
        Build the MPC reference over horizon H.
        State per step: [xc, yc, Lx, Ly].
        cmd_vel = [vx, vy]; for stepping in place pass [0, 0].
        Lateral sway alternates each step; sagittal travel rides on Ly_des.
        '''
        # sigma for the FIRST horizon step, keyed to current stance foot.
        sigma0 = +1.0 if stance_foot == "left_foot" else -1.0

        Ly_des = self.m * self.z_H * cmd_vel[0]          # forward command (0 for in-place)
        xc_des = 0.0                            # in-place: zero (scales with Ly otherwise)

        X_ref = np.zeros((4 * H,))
        for i in range(H):
            sigma = sigma0 * ((-1.0) ** i)      # flip each step across horizon
            yc_des, Lx_des = self.lateral_orbit_closed_form(sigma)
            X_ref[4 * i + 0] = xc_des
            X_ref[4 * i + 1] = yc_des
            X_ref[4 * i + 2] = Lx_des
            X_ref[4 * i + 3] = Ly_des

        print("ALIP DES:", X_ref[0:4])
        return X_ref
    
    def step_transition(self, x, u):
        return self.Ad @ (x + self.B @ u)

    def intra_step(self, x):
        return self.Ad_small @ x

    def solve_mpc(self, x0, cmd_vel, stance_foot):
        X_ref = self.make_X_ref(cmd_vel, stance_foot)
        b = self.phi @ x0 - X_ref
        P = 2 * self.gamma.T @ self.Q @ self.gamma
        q = 2 * self.gamma.T @ self.Q @ b

        scale = np.max(np.abs(P))
        P_scaled = P / scale
        q_scaled = q / scale

        # box bounds become lb/ub arrays instead of a list of tuples
        lb = np.array([b[0] for b in self.get_bounds(stance_foot)])
        ub = np.array([b[1] for b in self.get_bounds(stance_foot)])

        U = solve_qp(P_scaled, q_scaled, lb=lb, ub=ub, solver="quadprog")
        # print("MPC Results: ", U[:2])
        return U[:2]
    
#     def run_mpc(self, x0, v_x_des, v_y_des, steps=10):
#         curr_world_u = np.zeros(2)
#         com_traj = []
#         foot_traj = []
#         x = x0
#         for i in range(steps):
#             u = self.solve_mpc(x, v_x_des, v_y_des)
#             print(f"Step {i}: CoM state = {x}, footstep = {u}")
            
#             # record pre-transition CoM in world frame
#             com_traj.append(curr_world_u + x[:2])
            
#             # apply step transition
#             x = self.step_transition(x, u)
#             foot_traj.append(curr_world_u + u)
#             curr_world_u += u
            
#             # simulate intra-step dynamics
#             n_intra = int(self.T_s / self.dt_control)
#             for j in range(n_intra):
#                 x = self.intra_step(x)
#                 com_traj.append(curr_world_u + x[:2])
            
#         return np.array(com_traj), np.array(foot_traj)
    
#     def plot_steps(self, com_traj, foot_traj):
#         plt.figure()
#         plt.plot(com_traj[:, 0], com_traj[:, 1], 'b-', linewidth=1, alpha=0.5)
#         plt.scatter(com_traj[:, 0], com_traj[:, 1], label='CoM', marker='o', color='blue', s=10)
#         plt.scatter(foot_traj[:, 0], foot_traj[:, 1], label='Feet', marker='x', color='red', s=100)
#         for i in range(len(foot_traj)):
#             plt.annotate(str(i), foot_traj[i], textcoords="offset points", xytext=(5, -10), color='red', fontsize=8)
#         plt.xlabel('x (m)')
#         plt.ylabel('y (m)')
#         plt.legend()
#         plt.axis('equal')
#         plt.grid(True)
#         plt.show()

# if __name__ == "__main__":
#     T_s = 0.35

#     model = ALIP_MPC(T_s)

#     m, z_H, T_s, W = 30.0, 0.8, 0.35, 0.2   # your real values
#     cf = model.lateral_orbit_closed_form(m, z_H, T_s, W, sigma=+1.0)
#     ex = model.lateral_orbit_expm(m, z_H, T_s, W)
#     print("closed form (sigma=+1):", cf)
#     print("expm solve   (w=+W)   :", ex)



    