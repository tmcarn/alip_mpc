'''
Push force planning and hand normal-force regulation.

The push force is PLANNED (from the box estimates and the desired box acceleration), not taken
from a measurement, so the same value can be shared by the hands, the WBC force term and the
ALIP-MPC. The hands are position controlled, so the force is realized by choosing how deep
inside the face each hand target sits: an admittance loop adjusts the depth until the measured
normal force matches the planned one. The measurement is only used for this tracking.
'''
import numpy as np

from intention import HANDS

G = 9.81


class PushForceController:
    def __init__(self, dt, box_mass, box_mu, a_des=0.0,
                 K_est=500.0, k_p=0.0005, k_i=0.05, depth_max=0.06, tau_filt=0.02,
                 engage_frac=0.5, release_time=0.3):
        '''
        box_mass, box_mu: box estimates used to plan the force
        a_des:            desired box acceleration along the push (m/s^2), 0 = steady slide
        K_est:            rough hand-contact stiffness (N/m), only for the feedforward depth F/K
        k_p, k_i:         depth correction gains on the normal force error (m/N, m/(N s))
        depth_max:        clamp on the target depth inside the face (m)
        tau_filt:         low-pass time constant on the measured normal force (s), removes contact noise
        engage_frac:      a hand engages once its measured normal force reaches this fraction of F_des/2
        release_time:     ...and releases after this long with (almost) no force (s). ~one step, so the
                          per-step contact flicker doesn't toggle the planned force on and off
        '''
        self.dt = dt
        self.box_mass, self.box_mu, self.a_des = box_mass, box_mu, a_des
        self.K_est = K_est
        self.k_p, self.k_i = k_p, k_i
        self.depth_max = depth_max
        self.alpha = dt / (tau_filt + dt)

        self.F_n_meas = {h: 0.0 for h in HANDS}  # filtered measured normal force per hand (N, pushing > 0)
        self.integ = {h: 0.0 for h in HANDS}     # integral depth correction (m)
        self.depth = {h: self.F_des_hand / K_est for h in HANDS}

        # Engagement: the planned force is only applied (WBC, MPC) while a hand is on the box.
        # The measurement is used for this contact decision only, never as the force value.
        self.engage_frac, self.release_time = engage_frac, release_time
        self.engaged = {h: False for h in HANDS}
        self.t_no_force = {h: 0.0 for h in HANDS}

    @property
    def F_des(self):
        '''Planned total push force (N): overcome friction + accelerate the box.'''
        return self.box_mu * self.box_mass * G + self.box_mass * self.a_des

    @property
    def F_des_hand(self):
        return 0.5 * self.F_des  # split equally between the hands (symmetric contacts)

    def update(self, f_meas, n_world):
        '''
        f_meas:  {hand: measured contact force ON the robot (3,), world frame}
        n_world: {hand: outward box face normal at the contact (3,), world frame}
        Updates and returns {hand: target depth inside the face (m)}.
        '''
        F_ff = self.F_des_hand
        depth_ff = F_ff / self.K_est
        for h in HANDS:
            # the box pushes the robot back along its outward normal while the hand presses
            f_n = float(f_meas[h] @ n_world[h])
            self.F_n_meas[h] += self.alpha * (f_n - self.F_n_meas[h])

            e = F_ff - self.F_n_meas[h]                  # > 0: pressing too lightly -> go deeper
            integ_new = self.integ[h] + self.k_i * e * self.dt
            d = depth_ff + self.k_p * e + integ_new
            d_clipped = float(np.clip(d, 0.0, self.depth_max))
            if d == d_clipped:                            # anti-windup: only integrate when not saturated
                self.integ[h] = integ_new
            self.depth[h] = d_clipped

            # engage on clear contact, release only after ~a step of no contact (hysteresis)
            if self.F_n_meas[h] >= self.engage_frac * F_ff:
                self.engaged[h] = True
                self.t_no_force[h] = 0.0
            elif self.F_n_meas[h] < 0.1 * F_ff:
                self.t_no_force[h] += self.dt
                if self.t_no_force[h] >= self.release_time:
                    self.engaged[h] = False
        return self.depth

    def planned_hand_forces(self, n_world):
        '''
        {hand: planned force ON the robot (3,), world frame, or None if not engaged}
        = +n * F_des/2 (the box pushes back along its outward normal). Shared by the WBC and MPC.
        '''
        return {h: self.F_des_hand * np.asarray(n_world[h]) if self.engaged[h] else None for h in HANDS}
