# ALIP-MPC Humanoid: Walking and Box Pushing

A walking controller for a simulated humanoid (MuJoCo), built from scratch.
The robot can walk in any direction, turn, and push a box with both hands.

![Walking Demo](media/ALIP_MPC.gif)

## How it works (big picture)

Every control tick (1 ms) the controller does three things:

1. **Decide where to step.** The *ALIP-MPC* models the robot as an inverted pendulum
   and picks the next footstep so the robot keeps its balance and walks at the commanded speed.
2. **Make smooth paths.** The *trajectory generators* turn that footstep into a smooth
   path for the swing foot, and turn the yaw command into a smooth heading for the torso and feet.
3. **Turn goals into motor torques.** The *whole-body controller (WBC)* solves a small
   optimization that tracks all the goals at once (swing foot, torso, CoM height, hands)
   while respecting physics, foot friction and motor limits.

`run.py` ties these together and switches the stance foot after every step.

## Files

| File | What it does |
|------|--------------|
| `run.py`       | Main loop: reads the sim, runs the controllers, steps the sim |
| `alip_mpc.py`  | Footstep planner (ALIP model + MPC) |
| `swingtraj.py` | Swing foot path (position and yaw) |
| `torsotraj.py` | Torso heading (yaw) path for turning |
| `wbc.py`       | Whole-body controller: goals → joint torques |
| `intention.py` | Box geometry and where the hands should touch the box |
| `push.py`      | Push force planning and hand force tracking |
| `env.py`       | MuJoCo wrapper (robot state, box state, hand contact forces) |
| `constants.py` | Shared settings (step time, CoM height, box estimates, ...) |
| `debug_viz.py` | Debug markers in the viewer |
| `xml_files/`   | Robot, box and scene models |

## Setup and run

```bash
pip install -r requirements.txt
mjpython run.py     # mjpython is needed for the viewer on macOS
```

Change the walking command in `run.py` (`cmd_lin_vel` = forward/sideways speed,
`cmd_yaw_rate` = turning speed).

## Walking

### The ALIP model
The robot is treated as a point mass at constant height `z_H` balancing on the stance foot.
The state is the CoM position relative to the stance foot and the angular momentum about the foot:

```
x = [x_c, y_c, L_x, L_y]

ẋ_c =  L_y / (m z_H)        L̇_y =  m g x_c
ẏ_c = −L_x / (m z_H)        L̇_x = −m g y_c
```

Over one step of length `T_s` this becomes `x_end = A_d x_start` with `A_d = exp(A T_s)`.
When the foot lands at offset `u = [u_x, u_y]` from the old stance foot, the CoM position is
re-measured from the new foot:

```
x⁺ = x⁻ + B u,   B = [−I; 0]       (L is unchanged at impact)
```

### Footstep MPC
Plan `H = 4` steps ahead, use only the first one:

```
min_U  Σ_k ‖x_k − x_ref,k‖²_Q + r ‖U‖²      x_{k+1} = A_d x_k + B u_k
s.t.   step bounds (min step width, max reach)
```

The reference `x_ref` is the periodic walking orbit for the commanded speed:
- forward: `L_y,des = m z_H v_x`
- sideways sway: `y_c,des = ±W/2`, `L_x,des = ±½ m z_H ℓ W tanh(ℓ T_s/2)` with `ℓ = √(g/z_H)`, sign flips every step
- sideways walking: an offset `−m z_H v_y` on `L_x`

### Replanning mid-step
The MPC re-solves every 4 ms during the step (stopping at 85% of the step), so mistakes are
corrected before the foot lands. The measured state at time `t` is predicted to the end of the step:

```
x(T_s) = exp(A (T_s − t)) x(t)
```

### Robot's own frame
The MPC works in the stance foot's frame (x = forward, y = left), so "walk forward" means forward
for the robot, not for the world. With `R` the stance foot's yaw rotation:

```
in:   [x_c, y_c] = Rᵀ r,   [L_x, L_y] = Rᵀ L
out:  u_world = R u
```

### Turning
Each step the heading advances by `Δψ = ψ̇_cmd T_s`.
- **Torso:** ends each step halfway to the new heading, `ψ_torso → ψ_stance + ½ Δψ` (linear ramp).
- **Swing foot:** lands at the new heading, `ψ_foot → ψ_prev + Δψ` (cosine blend, so it lands with zero yaw speed).

### Whole-body controller
Solves for joint accelerations `q̈` and the stance foot force `λ`:

```
min   Σ_tasks w_i ‖J_i q̈ + J̇_i q̇ − a_i,des‖²       a_des = Kp (target − actual) + Kd (v_target − v)
s.t.  floating-base dynamics:   M q̈ + h = Sᵀτ + J_stᵀ λ + J_handᵀ f_hand
      stance foot doesn't move:  J_st q̈ + J̇_st q̇ = 0
      friction pyramid on λ, torque limits on τ
```

Tasks: swing foot position and orientation, torso orientation, CoM height, hand positions,
arm posture. The torques are then `τ = M q̈ + h − J_stᵀ λ − J_handᵀ f_hand`.

## Box pushing

### What works now
1. **Hand targets on the box.** Both hands are placed on the box face opposite the push
   direction, at the same height, centered so a straight push does not spin the box.
   The targets are stored in the box's frame (`c_h`, normal `n_h`), so they move and turn with the box:
   ```
   p_hand,des = p_box + R_box (c_h − depth · n_h)
   v_hand,des = v_box + ω_box × (p_hand,des − p_box)
   ```
2. **Planned push force.** The force needed to slide the box, using our *estimates* of the box
   (`BOX_MASS_EST`, `BOX_MU_EST` in `constants.py`). This one planned number is used everywhere
   below, instead of a noisy measurement:
   ```
   F_des = μ m_box g + m_box a_des        (split equally: F_des/2 per hand)
   ```
3. **Hands apply that force.** The hands are position controlled, so to push harder they aim
   slightly *inside* the box face. A small PI loop adjusts how deep, until the measured normal
   force matches the planned one:
   ```
   e = F_planned − F_measured
   depth = F_planned / K_est + k_p e + ∫ k_i e dt        (clamped to 0–6 cm)
   ```
   The measured force is only used for this check and to detect contact (a hand "engages" once
   it feels half its planned force, and releases after 0.3 s without force).
4. **WBC knows about the push.** The force on the robot from the box, `f_hand = +n · F_planned`,
   is added to the dynamics (see the WBC equations above), so the WBC isn't surprised by the box
   pushing back.
5. **MPC knows about the push.** The push creates a torque about the stance foot:
   ```
   τ = Σ_h (c_h − p_stance) × f_h       (only the x, y parts, in the stance frame)
   ```
   It enters the angular momentum equations, so each step gets a constant extra term:
   ```
   ẋ = A x + E τ        →     x⁺ = A_d x + B u + d_F,    d_F = ∫₀^{T_s} e^{As} ds · E τ
   ```
   It also shifts the walking orbit, so the robot leans into the box:
   ```
   Δx_c = −τ_y / (m g),   Δy_c = τ_x / (m g)        (forward push of force F at height h: Δx_c = h F / (m g))
   ```

### Next: box pose controller (to do)
Goal: push the box to a target position **and** heading, and make the robot walk along with it.

The hands can only push *into* the face, so they can make two things:
- a **forward force** (both hands push), and
- a **turning force** (one hand pushes harder than the other).

They can't push the box sideways. So the box is steered like a car:

1. **Steer.** Far from the goal, turn the box so its pushing direction `d` points at the goal.
   Close to the goal, switch to turning it to the goal heading:
   ```
   e_steer = angle(d) − angle(p_goal − p_box)
   e_goal  = ψ_box − ψ_goal
   e_ψ     = b e_steer + (1 − b) e_goal,     b = clip(distance / r_blend, 0, 1)
   ```
2. **Spring-damper on the box.** Pretend the box is attached to its goal by a spring and
   damper. This gives the force and turning effect the box needs. Limit the box speed
   (e.g. 0.15 m/s) and acceleration so it starts and stops smoothly:
   ```
   v_des = clip(K_v · (distance to go along d), 0, v_max)       (can only push: never negative)
   F     = μ m g + m a_des + D_v (v_des − v)
   M_z   = −K_ψ e_ψ − D_ψ ω_z
   ```
3. **Split between the hands.** Choose how hard each hand pushes so the total matches the
   forward force and the turning effect, while each hand keeps a small minimum force so it
   stays on the box. With `m_h = [(c_h − p_box) × d_h]_z` the turning effect per newton of hand `h`:
   ```
   f_r + f_l = F
   m_r f_r + m_l f_l = M_z
   f_min ≤ f_h ≤ f_max        (solve as a tiny QP when a limit is hit)
   ```
4. **Robot follows the box.** Keep the torso a fixed distance behind the hands, facing the box.
   The walking command is the speed of the hands plus a correction, turned into the robot's frame:
   ```
   p_torso,des = c_mid + standoff · n_face
   cmd_lin_vel = R_torsoᵀ [ v_contact + K_p (p_torso,des − p_torso) ]
   cmd_yaw_rate = ω_box + K_ψ,base (ψ_face − ψ_torso)
   ```
5. **Send the results on.** The hand forces go to the hand force loop, the WBC and the MPC.
   The walking commands go to the MPC and the heading planners.

Test in this order: straight push → goal off to the side (steering) → change of final heading.

A first draft (`BoxPoseController` in `push.py`) exists but does not work yet.

Things to keep in mind:
- The hands must be able to reach: shoulders are at about 1.8 m high and the arm is 0.84 m
  long, so push at about 1.35 m.
- The box slides before it tips if `μ h < half width` (`h` = push height).
- A big final heading change can't be done from one face. That needs moving the hands to
  another face (the high-level planner's job, below).

### Later: high-level planner
Follow *Where to Touch, How to Contact* (Xie et al.): a learned policy picks **where to
touch** the box and **where the box should go next**; the box pose controller above carries
it out. `intention.py` already has the interface (`ContactIntention`) and a simple
hand-written placeholder (`HeuristicIntentionPlanner`). The paper's success test:
position error < 5 cm and rotation distance `1 − ⟨q_box, q_goal⟩² < 0.03`.

## Issues solved
- ✅ ~~WBC lagged behind the stepping trajectory~~ → separate gains for each WBC task
- ✅ ~~Footsteps went out of control~~ → the MPC applied the step before the in-step motion; fixed the order (move first, then step)
- ✅ ~~First step unstable~~ → added a foot friction limit to the WBC
- ✅ ~~Position drifted at zero speed~~ → track the swing foot's target velocity; detect touchdown by foot height
- ✅ ~~Walking speed limited to ~0.1–0.3 m/s~~ → lowered the CoM height (now up to 0.5 m/s)
- ✅ ~~Poor foot placement while turning~~ → replan mid-step
- ✅ ~~Robot walked in world directions instead of its own~~ → MuJoCo stores the base's linear
  velocity in the world frame, Pinocchio expects the base frame; it is now rotated: `v_base = R_baseᵀ v_world`

## TODO
- Box pose controller (above)
- Detect touchdown from joint torques instead of foot height (works on uneven ground)

## Residual RL (future)
Combine this model-based controller with learning: besides adding corrections to the
actions, let the policy tune model settings such as CoM height `z_H`, step time `T_s`,
step limits and WBC gains/weights.

## Notes
- The Pinocchio model is built from the MuJoCo file (`pin.buildModelsFromMJCF`).
- MuJoCo and Pinocchio order quaternions differently (`[w,x,y,z]` vs `[x,y,z,w]`) and store
  the base velocity in different frames; both are converted in `env.py`.
