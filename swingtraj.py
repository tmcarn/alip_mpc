import numpy as np
from constants import *

class SwingTrajectory:
    def __init__(self, clearance=0.1):
        self.clearance = clearance
        self.start_pos = None
        self.target_pos = None

        self.T_s = STEP_DURATION
    
    def reset(self, start_pos, target_pos):
        # called at each step transition
        self.start_pos = start_pos.copy()
        self.target_pos = target_pos.copy()
    
    # def get_target(self, t):
    #     '''
    #     X-Y interpolation, with sinusoidal z pattern
    #     '''
    #     s = np.clip(t / self.T_s, 0, 1)
    #     xy = (1 - s) * self.start_pos[:2] + s * self.target_pos[:2]
    #     z = self.start_pos[2] + self.clearance * np.sin(np.pi * s)
    #     return np.array([xy[0], xy[1], z])

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
        For Velocity Tracking
        '''
        if t < 0 or t > self.T_s:
            return np.zeros(3)
        s = t / self.T_s
        dsdt = 1.0 / self.T_s
        dblend_ds = 0.5 * np.pi * np.sin(np.pi * s)      # matches the cosine-blend xy fix
        xy_vel = dblend_ds * dsdt * (self.target_pos[:2] - self.start_pos[:2])
        z_vel = self.clearance * np.pi * np.cos(np.pi * s) * dsdt
        return np.array([xy_vel[0], xy_vel[1], z_vel])