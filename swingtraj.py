import numpy as np
from constants import *
from torsotraj import Rz, yaw_from_R

class SwingTrajectory:
    def __init__(self, clearance=0.1, yaw_init=0.0):
        self.clearance = clearance
        self.start_pos = None
        self.target_pos = None

        self.T_s = STEP_DURATION

        # heading of the most recent landing target; each step adds yaw_rate_cmd * T_s
        # (stays in sync with TorsoYawTrajectory.yaw_frame + delta_yaw if both get the same cmd)
        self.target_yaw = yaw_init
        self.start_yaw = yaw_init

    def reset(self, start_pos, target_pos, R_start=None, yaw_rate_cmd=0.0):
        '''
        Called at each step transition.
        R_start:      measured swing foot rotation (3x3) at liftoff (None -> no yaw ramp)
        yaw_rate_cmd: commanded yaw rate (rad/s); landing yaw advances by yaw_rate_cmd * T_s
        '''
        self.start_pos = start_pos.copy()
        self.target_pos = target_pos.copy()

        # the foot lands one step's worth of turning past the previous landing heading
        self.target_yaw = self.target_yaw + yaw_rate_cmd * self.T_s

        if R_start is None:
            self.start_yaw = self.target_yaw
            return
        # unwrap measured yaw to be near the target (avoid ±pi jumps)
        meas = yaw_from_R(R_start)
        self.start_yaw = self.target_yaw + np.arctan2(np.sin(meas - self.target_yaw),
                                                      np.cos(meas - self.target_yaw))

    def get_target(self, t):
        '''
        Cosine-blended x-y interpolation (matches Gibson et al. virtual constraint
        reference) so lateral velocity -> 0 at touchdown, not constant swing speed.
        Sinusoidal z clearance.
        '''
        s = np.clip(t / self.T_s, 0, 1)
        blend = 0.5 * (1 - np.cos(np.pi * s))   # 0 at s=0, 1 at s=1, zero slope at both ends
        xy = (1 - blend) * self.start_pos[:2] + blend * self.target_pos[:2]
        z = self.start_pos[2] + self.clearance * np.sin(np.pi * s)
        return np.array([xy[0], xy[1], z])
    
    def get_velocity(self, t):
        '''
        For Velocity Tracking, Cosine-blended x-y interpolation (matches Gibson et al. virtual constraint
        reference) so lateral velocity -> 0 at touchdown, not constant swing speed.
        Sinusoidal z clearance.
        '''
        if t < 0 or t > self.T_s:
            return np.zeros(3)
        s = t / self.T_s
        dsdt = 1.0 / self.T_s
        dblend_ds = 0.5 * np.pi * np.sin(np.pi * s)      # matches the cosine-blend xy fix
        xy_vel = dblend_ds * dsdt * (self.target_pos[:2] - self.start_pos[:2])
        z_vel = self.clearance * np.pi * np.cos(np.pi * s) * dsdt
        return np.array([xy_vel[0], xy_vel[1], z_vel])

    def get_yaw(self, t):
        '''Cosine-blended yaw: reaches target_yaw at touchdown with zero yaw rate.'''
        s = np.clip(t / self.T_s, 0, 1)
        blend = 0.5 * (1 - np.cos(np.pi * s))
        return (1 - blend) * self.start_yaw + blend * self.target_yaw

    def get_rot_target(self, t):
        '''Desired swing foot rotation matrix (flat, yawed).'''
        return Rz(self.get_yaw(t))

    def get_ang_velocity(self, t):
        '''Desired swing foot angular velocity (world frame).'''
        if t < 0 or t > self.T_s:
            return np.zeros(3)
        s = t / self.T_s
        dblend_dt = 0.5 * np.pi * np.sin(np.pi * s) / self.T_s
        return np.array([0.0, 0.0, dblend_dt * (self.target_yaw - self.start_yaw)])
