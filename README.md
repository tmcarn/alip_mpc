# ALIP-MPC Bipedal Walker

A from-scratch ALIP-based MPC footstep planner with whole-body control (WBC)
for a 3D 5-DOF-per-leg biped, simulated in MuJoCo.

## Overview

The control stack has three layers:

1. **ALIP-MPC** (`alip_mpc.py`) — Angular Momentum Linear Inverted Pendulum
   model predictive control. Plans footstep locations over a horizon by
   tracking a desired angular-momentum periodic orbit, solved as a QP.
2. **Swing trajectory** (`swingtraj.py`) — generates the swing-foot reference
   path (linear in xy, sinusoidal lift in z) between footstep targets.
3. **Whole-body control** (`wbc.py`) — solves for joint accelerations and
   contact forces via QP subject to the floating-base dynamics and a stance
   contact constraint, then recovers joint torques by inverse dynamics.

A time-based finite-state machine (`run.py`) switches stance every `T_s`
seconds and ties the three layers together.

## Files

| File | Purpose |
|------|---------|
| `alip_mpc.py`  | ALIP model + MPC QP footstep planner |
| `wbc.py`       | Whole-body controller (QP inverse dynamics) |
| `swingtraj.py` | Swing-foot trajectory generator |
| `env.py`       | MuJoCo environment wrapper |
| `debug_viz.py` | Viewer debug geoms (foot frames, swing target, plan) |
| `constants.py` | Shared parameters |
| `run.py`       | Integrated walking loop |
| `xml_files/`   | MuJoCo MJCF model |

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
mjpython run.py     # mjpython needed for the passive viewer on macOS
```

## Issues
- ✅ ~~WBC is not correctly tracking the stepping trajectory, it lags behind~~
   - Solution: separate PD gains for each task of the WBC
- ✅ ~~Footstep locations become out of control~~
   - Solution: MPC State Transitions were incorrect, teleporting foot to new position first, before doing intrastep dynamics. As a result, foot was directed to go where it needed to instantly be, not where it should be by the time it gets there. 
- ✅ ~~First Step is Unstable~~
   - Solution: Added friction cone constraint to WBC
- ✅ ~~Position drifts when commanded velocity is zero~~
   - Still drifts but much less
   - Solutions: 
      - PD controller velocity was set to zero, set vel tracking to target velocity from the swing trajectory generator
      - Height Based Contact Detection Added

## TODO
- Switch height based contact detection to joint torque based (invariant to unlevel terrain)
- Modify X_ref to allow for non-zero velocity tracking, and turning
- Modify MPC formulation to include intra-step dynamics, and friction constraints for more dynamic manuvers


## Notes

- The Pinocchio model is built off of the MuJoCo
  directly via `pin.buildModelsFromMJCF`.
- MuJoCo and Pinocchio use different quaternion orderings
  (`[w,x,y,z]` vs `[x,y,z,w]`); configurations are converted at the boundary.