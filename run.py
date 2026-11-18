import numpy as np
import time
from constants import *
import pinocchio as pin

from env import BipedEnv

from swingtraj import SwingTrajectory
from torsotraj import TorsoYawTrajectory, Rz, yaw_from_R

from alip_mpc import ALIP_MPC
from wbc import WholeBodyController, WBCReference

from intention import BoxGeometry, HANDS, quat_wxyz_to_R
from push import PushForceController

from debug_viz import DebugVisualizer
from logger import Logger

class WalkingController:
    def __init__(self, model, data, dt, mpc, wbc, swing_planner, torso_yaw_planner):
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
        self.replan_every = max(1, int(round(REPLAN_PERIOD / self.dt)))  # ticks between mid-step re-solves
        self.phase_counter = self.steps_per_phase # initialized to force a switch immediately (stance = left, swing = right)

        self.mpc:ALIP_MPC = mpc
        self.wbc:WholeBodyController = wbc

        self.swing_planner:SwingTrajectory = swing_planner
        self.torso_yaw_planner:TorsoYawTrajectory = torso_yaw_planner

        self.cmd_lin_vel = np.array([0.5, 0.0])  # [forward, lateral] command velocity, body (stance foot) frame
        self.cmd_yaw_rate = 0.0  # yaw rate command (rad/s)

        self.final_swing_target = None
        self.swing_target = None
        self.com_pos = None

        # Hand targets fixed to the box: contact points in the BOX frame (HANDS order), set with
        # set_hand_contacts(), pressed push_depth into the face along the inward normal
        self.hand_contacts_obj = None    # (2, 3) contact points, box frame
        self.hand_normals_obj = None     # (2, 3) outward face normals, box frame
        self.hand_vel_ff = True          # feed the box's velocity forward (False -> v_des = 0, hands lag)

        # Planned push force; sets each hand's depth inside the face (normal-force admittance)
        self.push = PushForceController(dt, BOX_MASS_EST, BOX_MU_EST)

        # box state, updated every tick with set_box_state()
        self.box_pos, self.box_R = None, None
        self.box_vel, self.box_omega = np.zeros(3), np.zeros(3)

    def set_hand_contacts(self, points_obj, normals_obj):
        '''Contact points and outward normals (each (2, 3), HANDS order) in the box frame.'''
        self.hand_contacts_obj = np.asarray(points_obj, dtype=float)
        self.hand_normals_obj = np.asarray(normals_obj, dtype=float)

    def set_box_state(self, pos, R, vel, omega):
        '''Box pose and velocity, world frame (omega in world frame too).'''
        self.box_pos, self.box_R, self.box_vel, self.box_omega = pos, R, vel, omega

    def hand_normals_world(self):
        '''{hand: outward box face normal at its contact (3,), world frame}.'''
        return {h: self.box_R @ self.hand_normals_obj[i] for i, h in enumerate(HANDS)}

    def update_push_force(self, f_meas):
        '''Track the planned push force: adjust each hand's depth from its measured contact force.'''
        self.push.update(f_meas, self.hand_normals_world())

    def get_planned_hand_forces(self):
        '''{hand: planned force ON the robot (3,), world frame, or None if that hand isn't engaged}.'''
        return self.push.planned_hand_forces(self.hand_normals_world())

    def get_push_torque(self):
        '''
        Planned hand-force torque about the stance foot, stance foot frame, [tau_x, tau_y] for the
        ALIP-MPC: sum_h (c_h - p_stance) x f_h, with c_h the planned contact point on the box face.
        Uses self.R_stance_yaw, so call after get_ALIP_state(). Returns None when no hand is engaged.
        '''
        forces = self.get_planned_hand_forces()
        if all(f is None for f in forces.values()):
            return None
        p_st = self.get_foot_pos(self.stance_foot)
        tau = np.zeros(3)
        for i, h in enumerate(HANDS):
            if forces[h] is None:
                continue
            c_w = self.box_pos + self.box_R @ self.hand_contacts_obj[i]  # contact point, world
            tau += np.cross(c_w - p_st, forces[h])
        return (self.R_stance_yaw.T @ tau)[:2]

    def get_hand_reference(self):
        '''
        Hand targets attached to the box: p = p_box + R_box @ (c - depth * n), so they move and rotate
        with the box. The depth per hand comes from the push force controller.
        Velocity of a point rigidly attached to the box: v = v_box + w_box x (p - p_box).
        Returns {hand: (pos (3,), vel (3,) or None)}, world frame.
        '''
        refs = {}
        for i, hand in enumerate(HANDS):
            p_obj = self.hand_contacts_obj[i] - self.push.depth[hand] * self.hand_normals_obj[i]  # inside the face
            r = self.box_R @ p_obj                       # box center -> target, world frame
            pos = self.box_pos + r
            vel = self.box_vel + np.cross(self.box_omega, r) if self.hand_vel_ff else None
            refs[hand] = (pos, vel)
        return refs


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
        foot_id = self.pin_model.getFrameId(foot_name, pin.FrameType.BODY)  # BODY: sites share these names
        foot_pos = self.pin_data.oMf[foot_id].translation.copy()  # world frame position (3,)
        return foot_pos

    def get_foot_rot(self, foot_name):
        foot_id = self.pin_model.getFrameId(foot_name, pin.FrameType.BODY)  # BODY: sites share these names
        foot_rot = self.pin_data.oMf[foot_id].rotation.copy()  # world frame rotation (3, 3)
        return foot_rot

    def replan(self, q, dq, t_elapsed):
        '''Re-solve the MPC mid-step and move the swing foot's landing target.'''
        x = self.get_ALIP_state(q, dq)  # stance foot frame
        u_st = self.mpc.solve_mpc(x, self.cmd_lin_vel, self.stance_foot, t_elapsed=t_elapsed, verbose=False,
                                  tau_ext=self.get_push_torque())  # planned hand-force torque
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

        first_step = self.final_swing_target is None  # always plan on the first tick
        if touched_down or past_max or first_step:
        # if self.phase_counter >= self.steps_per_phase: # impact has occured
            self.phase_counter = 0

            # the foot that was swinging has landed -> becomes new stance foot and vice versa
            self.stance_foot, self.swing_foot = self.swing_foot, self.stance_foot


            # Only on impact, calculate next footstep location
            x = self.get_ALIP_state(q, dq)
            # print("ALIP STATE: ", x)
            u_st = self.mpc.solve_mpc(x, self.cmd_lin_vel, self.stance_foot,
                                      tau_ext=self.get_push_torque())  # stance foot frame, planned hand-force torque
            u = self.R_stance_yaw @ np.array([u_st[0], u_st[1], 0.0])          # -> world frame
            new_stance_pos = self.get_foot_pos(self.stance_foot)
            self.final_swing_target = new_stance_pos + u
            self.final_swing_target[2] = 0.05

            # self.final_swing_target = self.get_foot_pos(self.swing_foot) # Used for standing still
            self.swing_planner.reset(self.get_foot_pos(self.swing_foot), self.final_swing_target,
                                     self.get_foot_rot(self.swing_foot), self.cmd_yaw_rate) # (Initial Position , Final Position, Initial Rotation, Cmd Yaw Rate)
            self.torso_yaw_planner.reset(self.pin_data.oMf[self.torso_id].rotation, self.cmd_yaw_rate) # (Initial Rotation, Cmd Yaw Rate)

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
        hand_ref = self.get_hand_reference()  # hands on the box contact points, moving with the box
        hand_force = self.get_planned_hand_forces()  # planned (not measured) force, only while engaged
        ref = WBCReference(
            swing_pos=self.swing_target,
            swing_vel=self.swing_vel_target,
            swing_rot=self.swing_foot_rot_target,
            swing_omega=self.swing_foot_ang_vel_target,
            torso_rot=self.torso_yaw_target,
            torso_omega=self.torso_yaw_vel_target,
            right_hand_pos=hand_ref["right_hand"][0],
            right_hand_vel=hand_ref["right_hand"][1],
            right_hand_force=hand_force["right_hand"],
            left_hand_pos=hand_ref["left_hand"][0],
            left_hand_vel=hand_ref["left_hand"][1],
            left_hand_force=hand_force["left_hand"],
        )

        tau = self.wbc.compute_control(q, dq, ref, self.stance_foot)
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
    torso_yaw = TorsoYawTrajectory(yaw_init=0.0)
    # Arm posture to hold: the arm entries of q_ref (= ARM_SETPOINTS, set above)
    wbc = WholeBodyController(pin_model, pin_data, tau_limits=env.tau_limits, q_ref=q)

    controller = WalkingController(pin_model, pin_data, dt, mpc, wbc, swing, torso_yaw)

    def update_box_state():
        box_pos, box_quat, box_vel, box_omega_local = env.get_box_state()
        box_R = quat_wxyz_to_R(box_quat)
        # MuJoCo free joint: linear vel is world frame, angular vel is LOCAL (box) frame
        controller.set_box_state(box_pos, box_R, box_vel, box_R @ box_omega_local)
        return box_pos, box_R

    # Pick the hand contacts once: both hands on the face opposite the push direction,
    # symmetric about the push line through the box center (box frame, so they move with it)
    box_geom = BoxGeometry.from_mujoco(env.mj_model, "box_geom")
    box_pos, box_R = update_box_state()
    push_dir = np.array([1.0, 0.0])          # world push direction (x)
    pts, nrm, centered = box_geom.hand_push_points(push_dir, box_pos, box_R,
                                                   push_height=0.9, hand_spacing=0.5)
    controller.set_hand_contacts(pts, nrm)
    print("hand contacts (world):", box_pos + pts @ box_R.T, "centered:", centered)

    # print("START:", controller.get_foot_pos("left_foot"), controller.get_foot_pos("right_foot"))

    viz = DebugVisualizer(env.viewer)

    logger = Logger()

    
    for i in range(200000):
        q, dq = env.get_joint_state()
        pin.computeAllTerms(pin_model, pin_data, q, dq)
        pin.updateFramePlacements(pin_model, pin_data)
        update_box_state()

        # measured contact force: only used to track the planned push force (not for planning)
        f_hand = env.get_hand_contact_forces()  # force ON the robot, world frame
        controller.update_push_force(f_hand)

        tau = controller.compute_action(q, dq)
        env.step(tau)

        # --- push force tracking: planned vs measured normal force per hand ---
        if i % int(0.1 / dt) == 0:
            box_pos, _, box_vel, _ = env.get_box_state()
            p = controller.push
            # gap from each hand site to the box surface along the outward normal (> 0: not touching)
            n_w = controller.hand_normals_world()
            hand_ref = controller.get_hand_reference()
            gap, miss = {}, {}
            for i_h, h in enumerate(HANDS):
                p_hand = pin_data.oMf[wbc.right_hand_id if h == "right_hand" else wbc.left_hand_id].translation
                c_w = box_pos + controller.box_R @ controller.hand_contacts_obj[i_h]  # contact point on the face
                gap[h] = float((p_hand - c_w) @ n_w[h])
                miss[h] = float(np.linalg.norm(hand_ref[h][0] - p_hand))            # distance to target
            torso_x = pin_data.oMf[controller.torso_id].translation[0]
            print(f"t={i*dt:5.2f}  F_des/hand={p.F_des_hand:4.1f} N"
                  f"  Fn r/l={p.F_n_meas['right_hand']:5.1f}/{p.F_n_meas['left_hand']:5.1f} N"
                  f"  depth r/l={p.depth['right_hand']*100:4.1f}/{p.depth['left_hand']*100:4.1f} cm"
                  f"  gap r/l={gap['right_hand']*100:+5.1f}/{gap['left_hand']*100:+5.1f} cm"
                  f"  target miss r/l={miss['right_hand']*100:4.1f}/{miss['left_hand']*100:4.1f} cm"
                  f"  torso x={torso_x:.3f}  box x={box_pos[0]:.3f} vx={box_vel[0]:+.3f}"
                  f"  engaged r/l={int(p.engaged['right_hand'])}/{int(p.engaged['left_hand'])}")

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