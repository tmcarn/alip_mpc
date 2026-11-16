import numpy as np
import time
from constants import *
import pinocchio as pin

from env import BipedEnv

from swingtraj import SwingTrajectory
from torsotraj import TorsoYawTrajectory, Rz, yaw_from_R

from alip_mpc import ALIP_MPC
from wbc import WholeBodyController, WBCReference

from debug_viz import DebugVisualizer
from logger import Logger

class WalkingController:
    def __init__(self, model, data, dt, mpc, wbc, swing_planner, torso_yaw_planner):
        self.pin_model = model
        self.pin_data = data 

        self.feet = ["right_foot", "left_foot"]
        self.stance_foot = self.feet[1]
        self.stance_id = self.pin_model.getFrameId(self.stance_foot)

        self.swing_foot = self.feet[0]
        self.swing_id = self.pin_model.getFrameId(self.swing_foot)
        
        self.torso_id = self.pin_model.getFrameId("torso")

        self.T_s = STEP_DURATION  # step duration in seconds
        self.dt = dt  # control timestep

        self.steps_per_phase = int(self.T_s / self.dt)
        self.replan_every = max(1, int(round(REPLAN_PERIOD / self.dt)))  # ticks between mid-step re-solves
        self.phase_counter = self.steps_per_phase # initialized to force a switch immediately (stance = left, swing = right)

        self.mpc:ALIP_MPC = mpc
        self.wbc:WholeBodyController = wbc

        self.swing_planner:SwingTrajectory = swing_planner
        self.torso_yaw_planner:TorsoYawTrajectory = torso_yaw_planner   
        
        self.cmd_lin_vel = np.array([-0.3, 0.0])  # [forward, lateral] command velocity])
        self.cmd_yaw_rate = -0.1  # yaw rate command (rad/s)

        self.heading = 0.0  # current heading (yaw) of the robot in world frame
        self.des_heading = 0.0  # desired heading (yaw) of the robot in world frame

        self.final_swing_target = None
        self.swing_target = None
        self.com_pos = None


    def get_ALIP_state(self, q, dq):
        '''
        Map Robot Configuration to ALIP State [x_com, y_com, L_x, L_y],
        expressed in the stance foot frame: origin at the stance foot, axes rotated
        by the stance foot's measured yaw (x = foot forward, y = foot left, z = world up).
        '''
        # Compute CoM position and velocity
        pin.forwardKinematics(self.pin_model, self.pin_data, q, dq)
        pin.computeCentroidalMomentum(self.pin_model, self.pin_data, q, dq)
        pin.centerOfMass(self.pin_model, self.pin_data, q)
        com_pos = self.pin_data.com[0]
        self.com_pos = com_pos
        com_vel = self.pin_data.vcom[0]
        L_com = self.pin_data.hg.angular

        # Compute Lx, Ly using: L_contact = L_com + (r x m * v_com)
        m = self.pin_data.mass[0]  # total mass of robot

        # stance foot position
        stance_foot_pos = self.get_foot_pos(self.stance_foot)
        r = com_pos - stance_foot_pos  # vector from stance foot to CoM
        
        L_contact = L_com + np.cross(r, m * com_vel)  # total angular momentum about contact point

        # Rotate the world-aligned vectors into the stance foot frame (yaw only, so z stays
        # gravity-up and the planar ALIP model is unchanged). r and L are both vectors,
        # so they rotate the same way.
        R_stance = self.get_foot_rot(self.stance_foot)
        self.R_stance_yaw = Rz(yaw_from_R(R_stance))  # stance frame -> world (reused to map u back)
        r_st = self.R_stance_yaw.T @ r
        L_st = self.R_stance_yaw.T @ L_contact

        return np.concatenate([r_st[:2], L_st[:2]])  # x, y com position and Lx, Ly (contact angular momentum), stance foot frame

    def get_foot_pos(self, foot_name):
        foot_id = self.pin_model.getFrameId(foot_name)
        foot_pos = self.pin_data.oMf[foot_id].translation.copy()  # world frame position (3,)
        return foot_pos

    def get_foot_rot(self, foot_name):
            foot_id = self.pin_model.getFrameId(foot_name)
            foot_rot = self.pin_data.oMf[foot_id].rotation.copy()  # world frame rotation (3, 3)
            return foot_rot

    def replan(self, q, dq, t_elapsed):
        '''Re-solve the MPC mid-step and move the swing foot's landing target.'''
        x = self.get_ALIP_state(q, dq)  # stance foot frame
        u_st = self.mpc.solve_mpc(x, self.cmd_lin_vel, self.stance_foot, t_elapsed=t_elapsed, verbose=False)
        u = self.R_stance_yaw @ np.array([u_st[0], u_st[1], 0.0])  # -> world frame
        self.final_swing_target = self.get_foot_pos(self.stance_foot) + u
        self.final_swing_target[2] = 0.05
        self.swing_planner.update_target(self.final_swing_target)

    def compute_action(self, q, dq):
        ''' 
        Input: q, dq, and t, 
        Output: tau
        '''
        swing_pos = self.get_foot_pos(self.swing_foot)

        past_min = self.phase_counter >= int(MIN_FRAC * self.steps_per_phase)
        past_max = self.phase_counter >= int(MAX_FRAC * self.steps_per_phase)
        touched_down = past_min and swing_pos[2] <= HEIGHT_EPS

        if touched_down or past_max:
            self.phase_counter = 0

            # # Calculate Desired Heading based on commanded yaw rate, step duration, and current heading
            # self.des_heading += self.cmd_yaw_rate * self.T_s

            # the foot that was swinging has landed -> becomes new stance foot and vice versa
            self.stance_foot, self.swing_foot = self.swing_foot, self.stance_foot


            # Only on impact, calculate next footstep location
            x = self.get_ALIP_state(q, dq)
            print("ALIP STATE: ", x)
            u_st = self.mpc.solve_mpc(x, self.cmd_lin_vel, self.stance_foot)  # stance foot frame
            u = self.R_stance_yaw @ np.array([u_st[0], u_st[1], 0.0])          # -> world frame
            new_stance_pos = self.get_foot_pos(self.stance_foot)
            self.final_swing_target = new_stance_pos + u
            self.final_swing_target[2] = 0.05

            # self.final_swing_target = self.get_foot_pos(self.swing_foot) # Used for standing still
            self.swing_planner.reset(self.get_foot_pos(self.swing_foot), self.final_swing_target,
                                     self.pin_data.oMf[self.pin_model.getFrameId(self.swing_foot)].rotation, self.cmd_yaw_rate) # (Initial Position , Final Position, Initial Rotation, Cmd Yaw Rate)
            self.torso_yaw_planner.reset(self.pin_data.oMf[self.torso_id].rotation, self.cmd_yaw_rate) # (Initial Rotation, Cmd Yaw Rate)

            print("-" * 50)
            print("Torso Yaw Planner Reset: ")
            print("Start Yaw:", self.torso_yaw_planner.start_yaw)
            print("Target Yaw:", self.torso_yaw_planner.target_yaw)
            print("-" * 50)

        # --- mid-step re-solve (Gibson et al. Eq. 9) ---
        # Re-plan from the measured state, predicted to the end of the step, so model errors
        # during the step (e.g. while turning) are corrected before touchdown.
        elif (self.phase_counter % self.replan_every == 0
              and self.phase_counter < REPLAN_FREEZE_FRAC * self.steps_per_phase):
            self.replan(q, dq, self.phase_counter * self.dt)

        # print("Swing Foot Position:", self.get_foot_pos(self.swing_foot))
        # --- per-tick swing target ---
        t_phase = self.phase_counter * self.dt

        # Foot Position and Velocity Targets
        self.swing_target = self.swing_planner.get_target(t_phase)
        self.swing_vel_target = self.swing_planner.get_velocity(t_phase)

        # Foot Rotation and Angular Velocity Targets
        self.swing_foot_rot_target = self.swing_planner.get_rot_target(t_phase)
        self.swing_foot_ang_vel_target = self.swing_planner.get_ang_velocity(t_phase)

        # Torso Rotation and Angular Velocity Targets
        self.torso_yaw_target = self.torso_yaw_planner.get_target(t_phase)
        self.torso_yaw_vel_target = self.torso_yaw_planner.get_velocity(t_phase)

        # print("Swing Foot Target:", self.swing_target)

        # Caluclate tau, based on swing foot and swing target
        # --- WBC ---
        ref = WBCReference(
            swing_pos=self.swing_target,
            swing_vel=self.swing_vel_target,
            swing_rot=self.swing_foot_rot_target,
            swing_omega=self.swing_foot_ang_vel_target,
            torso_rot=self.torso_yaw_target,
            torso_omega=self.torso_yaw_vel_target,
        )

        tau = self.wbc.compute_control(q, dq, ref, self.stance_foot)
        tau = np.clip(tau, -ACTUATOR_LIMIT, ACTUATOR_LIMIT)

        self.phase_counter += 1

        return tau


def run():
    env = BipedEnv(XML_PATH)
    q, dq = env.reset()

    pin_model, _, _ = pin.buildModelsFromMJCF(XML_PATH)
    pin_data = pin_model.createData()

    pin.computeAllTerms(pin_model, pin_data, q, dq)
    pin.updateFramePlacements(pin_model, pin_data)

    dt = env.mj_model.opt.timestep
    T_s = STEP_DURATION

    mpc = ALIP_MPC(T_s, H=H, dt_control=dt)

    swing = SwingTrajectory(clearance=0.1)
    torso_yaw = TorsoYawTrajectory(yaw_init=0.0)

    wbc = WholeBodyController(pin_model, pin_data)

    controller = WalkingController(pin_model, pin_data, dt, mpc, wbc, swing, torso_yaw)

    print("START:", controller.get_foot_pos("left_foot"), controller.get_foot_pos("right_foot"))

    viz = DebugVisualizer(env.viewer)

    logger = Logger()

    
    for i in range(200000):
        q, dq = env.get_joint_state()
        pin.computeAllTerms(pin_model, pin_data, q, dq)
        pin.updateFramePlacements(pin_model, pin_data)

        tau = controller.compute_action(q, dq)
        env.step(tau)
        viz.draw(
            env.mj_data,
            None,                 # {"right_foot": id, "left_foot": id}
            swing_target=None, # yellow sphere (the current swing foot tracking target)
            footstep_plan=controller.final_swing_target, # magenta sphere (the MPC landing spot)
            com=controller.com_pos,
            torso_body_id=env.torso,
            draw_world=True
        )

        env.render()
        time.sleep(dt*1)

    env.close()



if __name__ == "__main__":
    run()