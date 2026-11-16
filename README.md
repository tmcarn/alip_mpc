# ALIP-MPC Bipedal Walker

A from-scratch ALIP-based MPC footstep planner with whole-body control (WBC)
for a 3D 5-DOF-per-leg biped, simulated in MuJoCo.

![Walking Demo](media/ALIP_MPC.gif)

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
   - Solutions: 
      - PD controller velocity was set to zero, set vel tracking to target velocity from the swing trajectory generator
      - Height Based Contact Detection Added
- ✅ ~~Velocity tracking breaks down at about 0.1 m/s for lateral walking and 0.3 m/s for forwards walking~~
   - Solution: Reduced nominal center of mass height (z_H) from 1.3m to 1.2m
   - Velocity tracking increased to 0.5 m/s
- ✅ ~~Feet are turning to track commanded angular velocity but foot placements are poor~~
   - Issue: MPC assumes linear trajectory, and since it only planned once per footstep, plans were always out of date
   - Solution: Include intra-step dynamics, to continously replan mid-step
- ✅ ~~Robot turns correctly but maintains command velocity in the world frame not the local frame~~
   - Solution: pass in local frame velocity into MPC, not global frame

## TODO
- Switch height based contact detection to joint torque based (invariant to unlevel terrain)

## Loco-Manipulation: Box Pushing (planned)
Given a contact position on the box and a target box pose, push the box there. Approach phase is assumed complete for now (robot starts at the pre-push pose).

### Push geometry
- Push direction: `d = (p_target - p_box) / |p_target - p_box|` (horizontal)
- Pure translation: push line must pass through the box CoM (contact on the face opposite `d`). A sideways contact offset induces yaw; steer by shifting contact laterally or with unequal hand forces, closed-loop on box yaw error
- Current box (1 m cube, 5 kg, mu = 0.5): sliding needs `F ~ mu*m*g ~ 25 N` (plan 30-35 N). Slides before tipping if `mu*h < 0.5 m`, push at ~0.5-0.7 m height

### Phases
| Phase | Walking cmd | Hands |
|---|---|---|
| Approach | P-control to pre-push pose (heading along `d`, standoff = reach + margin), error in body frame -> `cmd_lin_vel`, `cmd_yaw_rate` | Arm posture PD |
| Reach | Step in place | Hand sites to contact points, slightly inside the face |
| Push | Slow walk along `d` (0.1-0.2 m/s) + heading correction | Track contact points on the box (from box pose, every tick) |
| Release | Stop at pose tolerance, step back | Back to posture |

### Controller changes
1. **Hand task in WBC:** position task on `right_hand` / `left_hand` (Pinocchio `OP_FRAME`), replaces arm posture PD during Reach/Push
2. **Contact force:** impedance, target `delta` inside the surface so `F ~ K_hand * delta` (e.g. K = 900, delta ~ 4 cm -> ~35 N). Later: explicit hand-force variable in the QP
3. **Push force in WBC dynamics:** add `J_hand^T f_hand` to the floating-base rows, otherwise tasks fight an unmodeled ~30 N force
4. **Push force in ALIP:** hand force adds `(p_hand - p_stance) x f_hand` to `L_dot` (forward push: `Ly_dot += -z_h F`). Add as affine term `x+ = A_d x + B u + d_F` so the MPC leans into the box (CoM offset ~ `F z_h / (m g)`)

## High-Level Planner (future)
Hierarchical RL-MPC following *Where to Touch, How to Contact* (Xie et al.): RL predicts a **contact intention** (contact point(s) on the object + post-contact subgoal pose), a low-level MPC realizes it through contact.

| Layer | Paper | This stack |
|---|---|---|
| High level (where to touch) | RL -> contact keypoint(s) + subgoal | Same, ~0.2-1 Hz |
| Between contacts | Lift, move in free space, approach | Walk to pre-contact pose (Approach/Reach) |
| Low level (how to contact) | ComFree-MPC on object + end-effector | Box + hands manipulation MPC -> hand targets + base velocity, ~100 Hz |
| Execution | Arm impedance control (OSC) | WBC hand impedance tasks + ALIP-MPC / yaw planners for legs |

- Policy is embodiment-agnostic (object-frame contacts + subgoals): train in a box-only sim with 1-2 abstract point pushers, deploy on the humanoid without simulating walking during RL
- Hand impedance matches the paper's end-effector model (force = stiffness x commanded displacement)

### Build order
1. `ContactIntention` interface: contact points (box frame) + subgoal pose (or position/orientation cost weights)
2. Heuristic high level (placeholder): contact on face opposite target direction, subgoal = target capped to a max step
3. Push executor: intention -> hand targets, `cmd_lin_vel`, `cmd_yaw_rate`. Base follows hands: box velocity + gain on hand offset from nominal reach (body frame). Later: ComFree-style MPC
4. Push-force term in ALIP prediction (`d_F`), since the robot is balancing (no counterpart in the paper)
5. Abstract training env (box + floor + point pushers) with object-frame observations (keypoints, goal flow, clearance)

### Design choices
- 1 vs 2 contacts: pick one face keypoint and place both hands symmetric about it (`n = 1`), avoids the combinatorial action space
- Keypoints on the 4 vertical faces, collision term excludes the bottom / wall-blocked faces
- Penalize contact switches (each one means walking around the box) or include base pose relative to the box in the observation


## Residual RL
- Tie this model based controller into previous work with Residual RL
- Instead of just adding residuals to the action space, also modify upstream model parameters
   - z_H (Nominal CoM Height)
   - Ts (Nominal Step Duration)
   - Bounds on Step Locations?
   - WBC Params (Kp's, Kd's, Task Weights)?

## Notes

- The Pinocchio model is built off of the MuJoCo
  directly via `pin.buildModelsFromMJCF`.
- MuJoCo and Pinocchio use different quaternion orderings
  (`[w,x,y,z]` vs `[x,y,z,w]`); configurations are converted at the boundary.