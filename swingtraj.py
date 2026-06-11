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
    
    def get_target(self, t):
        '''
        X-Y interpolation, with sinusoidal z pattern
        '''
        s = np.clip(t / self.T_s, 0, 1)
        xy = (1 - s) * self.start_pos[:2] + s * self.target_pos[:2]
        z = self.start_pos[2] + self.clearance * np.sin(np.pi * s)
        return np.array([xy[0], xy[1], z])