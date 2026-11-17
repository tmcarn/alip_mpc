import numpy as np
from constants import *


def Rz(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def yaw_from_R(R):
    return np.arctan2(R[1, 0], R[0, 0])


class TorsoYawTrajectory:
    '''
    Torso yaw reference for heading control (Gibson et al. 2022, Eq. 19 hip-yaw rows).

    At each impact:
        dpsi      = yaw_rate_cmd * T_s
        psi_frame = heading of the new stance foot (frozen for the step; use for MPC frame)
        target    = psi_frame + 0.5 * dpsi   (torso ends midway between stance and next foot)
    Within the step the torso yaw ramps linearly from its measured value at impact to target,
    so in steady turning the torso yaw rate equals the commanded yaw rate.
    '''
    def __init__(self, yaw_init=0.0):
        self.T_s = STEP_DURATION
        self.yaw_frame = yaw_init  # heading of the current stance foot / MPC frame
        self.delta_yaw = 0.0           # heading change planned for this step
        self.start_yaw = yaw_init
        self.target_yaw = yaw_init

    def reset(self, R_torso, yaw_rate_cmd):
        '''
        Called at each step transition.
        R_torso: measured torso rotation (3x3) at impact.
        '''
        # the foot that just landed was aimed at yaw_frame + delta_yaw -> new stance heading
        self.yaw_frame = self.yaw_frame + self.delta_yaw
        self.delta_yaw = yaw_rate_cmd * self.T_s

        self.target_yaw = self.yaw_frame + 0.5 * self.delta_yaw
        # unwrap measured yaw to be near the target (avoid ±pi jumps)
        meas = yaw_from_R(R_torso)
        self.start_yaw = self.target_yaw + np.arctan2(np.sin(meas - self.target_yaw),
                                                      np.cos(meas - self.target_yaw))

    def get_yaw(self, t):
        s = np.clip(t / self.T_s, 0, 1)
        return (1 - s) * self.start_yaw + s * self.target_yaw

    def get_target(self, t):
        '''Desired torso rotation matrix (upright, yawed).'''
        return Rz(self.get_yaw(t))

    def get_velocity(self, t):
        '''Desired torso angular velocity (world frame).'''
        if t < 0 or t > self.T_s:
            return np.zeros(3)
        return np.array([0.0, 0.0, (self.target_yaw - self.start_yaw) / self.T_s])

    def get_swing_foot_target(self):
        '''Yaw the swing foot should land with (new heading).'''
        return Rz(self.yaw_frame + self.delta_yaw)
