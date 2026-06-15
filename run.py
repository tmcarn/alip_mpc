import numpy as np
import time
from constants import *
import pinocchio as pin

from env import BipedEnv
from swingtraj import SwingTrajectory
from alip_mpc import ALIP_MPC
from wbc import WholeBodyController

from debug_viz import DebugVisualizer

class WalkingController:
    def __init__(self, model, data, dt, mpc, wbc, swing_planner):
        self.pin_model = model
        self.pin_data = data 

        self.feet = ["right_foot", "left_foot"]
        self.stance_foot = self.feet[0]
        self.stance_id = self.pin_model.getFrameId(self.stance_foot)

        self.swing_foot = self.feet[1]
        self.swing_id = self.pin_model.getFrameId(self.swing_foot)
        
        self.torso_id = self.pin_model.getFrameId("torso")

        self.T_s = STEP_DURATION  # step duration in seconds
        self.dt = dt  # control timestep

        self.steps_per_phase = int(self.T_s / self.dt)
        self.phase_counter = self.steps_per_phase # initialized so Step Planner is kicked off immediately

        self.mpc:ALIP_MPC = mpc
        self.swing_planner:SwingTrajectory = swing_planner
        self.wbc:WholeBodyController = wbc

        self.cmd_vel = np.array([0, 0])

        self.final_swing_target = None
        self.swing_target = None



    def get_ALIP_state(self, q, dq):
            '''
            Map Robot Configuration to ALIP State [x_com, y_com, L_x, L_y]
            '''
            # Compute CoM position and velocity
            pin.forwardKinematics(self.pin_model, self.pin_data, q, dq)
            pin.centerOfMass(self.pin_model, self.pin_data, q)
            com_pos = self.pin_data.com[0]
            com_vel = self.pin_data.vcom[0]

            # Compute CoM Angular Momentum
            pin.computeCentroidalMomentum(self.pin_model, self.pin_data, q, dq)
            L_com = self.pin_data.hg.angular

            # Compute Lx, Ly using: L_contact = L_com + (r x m * v_com)
            m = self.pin_data.mass[0]  # total mass of robot

            # stance foot position
            stance_foot_pos = self.get_foot_pos(self.stance_foot)
            r = com_pos - stance_foot_pos  # vector from stance foot to CoM
            
            L_contact = L_com + np.cross(r, m * com_vel)  # total angular momentum about contact point

            return np.concatenate([r[:2], L_contact[:2]])  # x, y com position (w.r.t. stance foot) and Lx, Ly (contact angular momentum)

    def get_foot_pos(self, foot_name):
        foot_id = self.pin_model.getFrameId(foot_name)
        foot_pos = self.pin_data.oMf[foot_id].translation.copy()  # world frame position (3,)
        return foot_pos
    
    def compute_action(self, q, dq):
        ''' 
        Input: q, dq, and t, 
        Output: tau
        '''
        if self.phase_counter >= self.steps_per_phase: # impact has occured
            self.phase_counter = 0
            # the foot that was swinging has landed -> becomes new stance foot and vice versa
            self.stance_foot, self.swing_foot = self.swing_foot, self.stance_foot

            print(f"\n SWING: {self.swing_foot} \t STANCE: {self.stance_foot} \n")

            # Only on impact, calculate next footstep location
            x = self.get_ALIP_state(q, dq)
            u = self.mpc.solve_mpc(x, self.cmd_vel, self.stance_foot)
            new_stance_pos = self.get_foot_pos(self.stance_foot)
            self.final_swing_target = new_stance_pos + np.array([u[0], u[1], 0.0])
            self.final_swing_target[2] = 0.025

            # self.final_swing_target = self.get_foot_pos(self.swing_foot) # Used for standing still
            self.swing_planner.reset(self.get_foot_pos(self.swing_foot), self.final_swing_target) # (Initial Position , Final Position)
        
        print("Swing Foot Position:", self.get_foot_pos(self.swing_foot))
        # --- per-tick swing target ---
        t_phase = self.phase_counter * self.dt
        self.swing_target = self.swing_planner.get_target(t_phase)
        print("Swing Foot Target:", self.swing_target)

        # Caluclate tau, based on swing foot and swing target
        # --- WBC ---
        tau = self.wbc.compute_control(q, dq, self.swing_target, self.stance_foot)
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
    wbc = WholeBodyController(pin_model, pin_data)

    controller = WalkingController(pin_model, pin_data, dt, mpc, wbc, swing)

    print("START:", controller.get_foot_pos("left_foot"), controller.get_foot_pos("right_foot"))

    viz = DebugVisualizer(env.viewer)

    
    for i in range(20000):
        q, dq = env.get_joint_state()
        pin.computeAllTerms(pin_model, pin_data, q, dq)
        pin.updateFramePlacements(pin_model, pin_data)

        tau = controller.compute_action(q, dq)
        env.step(tau)
        viz.draw(
            env.mj_data,
            env.foot_body,                 # {"right_foot": id, "left_foot": id}
            swing_target=controller.swing_target,     # yellow sphere
            footstep_plan=controller.final_swing_target,          # magenta sphere (the MPC landing spot)
        )
        env.render()
        time.sleep(dt*10)

    env.close()


if __name__ == "__main__":
    run()