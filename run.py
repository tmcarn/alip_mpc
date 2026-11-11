import numpy as np
import time
from constants import *
import pinocchio as pin

from env import BipedEnv
from swingtraj import SwingTrajectory
from alip_mpc import ALIP_MPC
from wbc import WholeBodyController

from debug_viz import DebugVisualizer
from logger import Logger

class WalkingController:
    def __init__(self, model, data, dt, mpc, wbc, swing_planner):
        self.pin_model = model
        self.pin_data = data 

        self.feet = ["right_foot", "left_foot"]
        self.stance_foot = self.feet[1]
        self.stance_id = self.pin_model.getFrameId(self.stance_foot, pin.FrameType.BODY)

        self.swing_foot = self.feet[0]
        self.swing_id = self.pin_model.getFrameId(self.swing_foot, pin.FrameType.BODY)
        
        self.torso_id = self.pin_model.getFrameId("torso", pin.FrameType.BODY)

        self.T_s = STEP_DURATION  # step duration in seconds
        self.dt = dt  # control timestep

        self.steps_per_phase = int(self.T_s / self.dt)
        self.phase_counter = self.steps_per_phase # initialized to force a switch immediately (stance = left, swing = right)

        self.mpc:ALIP_MPC = mpc
        self.swing_planner:SwingTrajectory = swing_planner
        self.wbc:WholeBodyController = wbc

        self.cmd_vel = np.array([0.5, 0.0])  # [forward, lateral] command velocity])

        self.final_swing_target = None
        self.swing_target = None
        self.com_pos = None


    def get_ALIP_state(self, q, dq):
        '''
        Map Robot Configuration to ALIP State [x_com, y_com, L_x, L_y]
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

        return np.concatenate([r[:2], L_contact[:2]])  # x, y com position (w.r.t. stance foot) and Lx, Ly (contact angular momentum)

    def get_foot_pos(self, foot_name):
        foot_id = self.pin_model.getFrameId(foot_name, pin.FrameType.BODY)  # BODY: sites share these names
        foot_pos = self.pin_data.oMf[foot_id].translation.copy()  # world frame position (3,)
        return foot_pos
    
    def compute_action(self, q, dq):
        ''' 
        Input: q, dq, and t, 
        Output: tau
        '''
        swing_pos = self.get_foot_pos(self.swing_foot)

        past_min = self.phase_counter >= int(MIN_FRAC * self.steps_per_phase)
        past_max = self.phase_counter >= int(MAX_FRAC * self.steps_per_phase)
        touched_down = past_min and swing_pos[2] <= HEIGHT_EPS

        first_step = self.final_swing_target is None  # always plan on the first tick
        if touched_down or past_max or first_step:
        # if self.phase_counter >= self.steps_per_phase: # impact has occured
            self.phase_counter = 0

            # the foot that was swinging has landed -> becomes new stance foot and vice versa
            self.stance_foot, self.swing_foot = self.swing_foot, self.stance_foot


            # Only on impact, calculate next footstep location
            x = self.get_ALIP_state(q, dq)
            print("ALIP STATE: ", x)
            u = self.mpc.solve_mpc(x, self.cmd_vel, self.stance_foot)
            new_stance_pos = self.get_foot_pos(self.stance_foot)
            self.final_swing_target = new_stance_pos + np.array([u[0], u[1], 0.0])
            self.final_swing_target[2] = 0.05

            # self.final_swing_target = self.get_foot_pos(self.swing_foot) # Used for standing still
            self.swing_planner.reset(self.get_foot_pos(self.swing_foot), self.final_swing_target) # (Initial Position , Final Position)
        
        # print("Swing Foot Position:", self.get_foot_pos(self.swing_foot))
        # --- per-tick swing target ---
        t_phase = self.phase_counter * self.dt
        self.swing_target = self.swing_planner.get_target(t_phase)
        self.swing_vel_target = self.swing_planner.get_velocity(t_phase)
        # print("Swing Foot Target:", self.swing_target)

        # Caluclate tau, based on swing foot and swing target
        # --- WBC ---
        tau = self.wbc.compute_control(q, dq, self.swing_target, self.swing_vel_target, self.stance_foot)
        tau = np.clip(tau, -self.wbc.tau_limits, self.wbc.tau_limits)

        self.phase_counter += 1

        return tau


def run():
    env = BipedEnv(SCENE_PATH)
    q, dq = env.reset()
    q, dq = env.set_joint_angles(ARM_SETPOINTS)  # start the arms at their setpoints (no initial swing)

    pin_model, _, _ = pin.buildModelsFromMJCF(ROBOT_PATH)  # robot only (no box)
    pin_data = pin_model.createData()

    pin.computeAllTerms(pin_model, pin_data, q, dq)
    pin.updateFramePlacements(pin_model, pin_data)

    dt = env.mj_model.opt.timestep
    T_s = STEP_DURATION

    mpc = ALIP_MPC(T_s, H=H, dt_control=dt)
    swing = SwingTrajectory(clearance=0.1)
    # Arm posture to hold: the arm entries of q_ref (= ARM_SETPOINTS, set above)
    wbc = WholeBodyController(pin_model, pin_data, tau_limits=env.tau_limits, q_ref=q)

    controller = WalkingController(pin_model, pin_data, dt, mpc, wbc, swing)

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