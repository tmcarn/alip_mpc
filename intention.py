'''
Contact intention interface (Xie et al., "Where to Touch, How to Contact", Eq. 1).

A contact intention is the high-level action: WHERE to touch the object (one contact point
per hand on its surface, object frame) and WHAT the object should do after contact (a
subgoal pose in SE(3) = position + SO(3) orientation). Everything below it (push executor, WBC, ALIP-MPC)
consumes only this, so the heuristic planner here can later be swapped for an RL policy
with the same output.

Orientations are 3x3 rotation matrices (object -> world). Use quat_wxyz_to_R for MuJoCo
quaternions, e.g. from env.get_box_state().
'''
import numpy as np
from dataclasses import dataclass
from typing import Optional

import pinocchio as pin

HANDS = ("right_hand", "left_hand")  # contact order in a ContactIntention (Pinocchio frame names)


def quat_wxyz_to_R(q):
    '''MuJoCo (w, x, y, z) quaternion -> rotation matrix.'''
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def R_to_quat_wxyz(R):
    q = pin.Quaternion(np.asarray(R, dtype=float))  # Pinocchio coeffs are (x, y, z, w)
    x, y, z, w = q.coeffs()
    return np.array([w, x, y, z])


def rotation_distance(R_a, R_b):
    '''Paper's rotation distance 1 - <q_a, q_b>^2 (0 = same orientation, 1 = 180 deg apart).'''
    return float(1.0 - np.dot(R_to_quat_wxyz(R_a), R_to_quat_wxyz(R_b)) ** 2)


@dataclass
class ContactIntention:
    '''
    contact_points:  (2, 3) contact locations on the object surface, OBJECT frame,
                     one per hand in HANDS order: [right_hand, left_hand]
    contact_normals: (2, 3) outward surface normals at those points, object frame
                     (each hand presses along -normal)
    subgoal_pos:     (3,)  post-contact object position, world frame
    subgoal_R:       (3,3) post-contact object orientation (object -> world), SO(3)
    keypoint_ids:    (2,) indices into BoxGeometry.keypoints (what a discrete RL policy
                     outputs); None if the points are continuous
    w_pos, w_ori:    subgoal cost weights (paper's Option 1). w_ori = 0 -> position-only subgoal
    '''
    contact_points: np.ndarray
    contact_normals: np.ndarray
    subgoal_pos: np.ndarray
    subgoal_R: np.ndarray
    keypoint_ids: Optional[np.ndarray] = None
    w_pos: float = 1.0
    w_ori: float = 0.0

    def __post_init__(self):
        self.contact_points = np.atleast_2d(np.asarray(self.contact_points, dtype=float))
        self.contact_normals = np.atleast_2d(np.asarray(self.contact_normals, dtype=float))
        self.subgoal_pos = np.asarray(self.subgoal_pos, dtype=float)
        self.subgoal_R = np.asarray(self.subgoal_R, dtype=float)
        assert self.contact_points.shape == (len(HANDS), 3), "one contact per hand"
        assert self.contact_normals.shape == self.contact_points.shape
        assert self.subgoal_R.shape == (3, 3)

    def contacts_world(self, obj_pos, obj_R):
        '''Contact points and outward normals (each (2, 3), HANDS order), world frame.'''
        return obj_pos + self.contact_points @ obj_R.T, self.contact_normals @ obj_R.T

    def hand_contacts_world(self, obj_pos, obj_R):
        '''{hand name: (contact point (3,), outward normal (3,))}, world frame.'''
        pts, nrm = self.contacts_world(obj_pos, obj_R)
        return {h: (pts[i], nrm[i]) for i, h in enumerate(HANDS)}

    def subgoal_error(self, obj_pos, obj_R):
        '''
        (position error (3,), orientation error (3,)) from the object to the subgoal, world frame.
        The orientation error is the rotation vector log(R_sub R_obj^T).
        '''
        return self.subgoal_pos - obj_pos, pin.log3(self.subgoal_R @ obj_R.T)

    def cost(self, obj_pos, obj_R):
        '''Terminal cost V (paper Eq. 3): weighted position + orientation error.'''
        e_pos = self.subgoal_pos - obj_pos
        return self.w_pos * float(e_pos @ e_pos) + self.w_ori * rotation_distance(obj_R, self.subgoal_R)

    def reached(self, obj_pos, obj_R, pos_tol=0.05, rot_tol=0.03):
        '''Paper's success test: position within pos_tol (m), rotation distance within rot_tol.'''
        pos_ok = np.linalg.norm(self.subgoal_pos - obj_pos) < pos_tol
        rot_ok = self.w_ori == 0.0 or rotation_distance(obj_R, self.subgoal_R) < rot_tol
        return pos_ok and rot_ok


class BoxGeometry:
    '''
    Box object geometry in its own frame (origin at the box center).
    Keypoints (the discrete contact candidates a policy picks from) are sampled on all six
    faces. Whether a face is pushable depends on the current orientation (a face pointing
    up or down is not), see pushable_faces().
    '''
    def __init__(self, half_extents, n_per_axis=5, face_margin=0.1):
        self.half = np.asarray(half_extents, dtype=float)
        self.face_margin = face_margin  # keep contacts this far from face edges (m)
        pts, nrm = [], []
        for axis in range(3):
            t1, t2 = [a for a in range(3) if a != axis]
            for sign in (+1.0, -1.0):
                n = np.zeros(3); n[axis] = sign
                for s1 in np.linspace(-1, 1, n_per_axis):
                    for s2 in np.linspace(-1, 1, n_per_axis):
                        p = np.zeros(3)
                        p[axis] = sign * self.half[axis]
                        p[t1] = s1 * max(self.half[t1] - face_margin, 0.0)
                        p[t2] = s2 * max(self.half[t2] - face_margin, 0.0)
                        pts.append(p); nrm.append(n)
        self.keypoints = np.array(pts)
        self.normals = np.array(nrm)

    @classmethod
    def from_mujoco(cls, mj_model, geom_name="box_geom", **kw):
        import mujoco
        gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        return cls(mj_model.geom_size[gid].copy(), **kw)

    def nearest_keypoint(self, p_obj):
        return int(np.argmin(np.linalg.norm(self.keypoints - p_obj, axis=1)))

    @staticmethod
    def pushable_faces(obj_R, max_tilt=np.radians(45)):
        '''Object-frame face axes whose world normal is within max_tilt of horizontal.'''
        up_obj = obj_R.T @ np.array([0.0, 0.0, 1.0])
        return [a for a in range(3) if abs(up_obj[a]) < np.sin(max_tilt)]

    def push_point(self, d_world, obj_pos, obj_R, push_height):
        '''
        Contact point whose push line (horizontal, along d_world, at push_height above the
        ground) passes through the box center in plan view, so the push translates without
        yawing the box. Returns (point, outward normal) in the object frame.
        '''
        d = np.array([d_world[0], d_world[1], 0.0])
        d /= np.linalg.norm(d)
        d_obj = obj_R.T @ d
        faces = self.pushable_faces(obj_R)
        axis = max(faces, key=lambda a: abs(d_obj[a]))      # face the push ray exits through
        # ray p(t) = p0 - d_obj * t, starting above/below the center at the push height,
        # intersected with the face plane p[axis] = -s * half[axis]
        s = np.sign(d_obj[axis])
        p0 = obj_R.T @ np.array([0.0, 0.0, push_height - obj_pos[2]])
        t = (p0[axis] + s * self.half[axis]) / d_obj[axis]
        p = p0 - d_obj * t
        p[axis] = -s * self.half[axis]                       # exactly on the face plane (rounding)
        for a in range(3):                                   # stay on the face, off the edges
            if a != axis:
                lim = max(self.half[a] - self.face_margin, 0.0)
                p[a] = np.clip(p[a], -lim, lim)
        n = np.zeros(3); n[axis] = -np.sign(d_obj[axis])
        return p, n

    def hand_push_points(self, d_world, obj_pos, obj_R, push_height, hand_spacing):
        '''
        Two contacts on the same face, hand_spacing apart horizontally and symmetric about
        push_point() so the net push still goes through the center. Returns
        (points (2, 3), normals (2, 3), centered) in the object frame, HANDS order.
        Right hand = robot's right while facing into the face (pushing along -normal).
        If the pair would run off the face, both are shifted together and centered=False
        (the push then also yaws the box).
        '''
        p, n = self.push_point(d_world, obj_pos, obj_R, push_height)
        n_w = obj_R @ n
        fwd = -n_w[:2] / np.linalg.norm(n_w[:2])             # robot faces into the face
        right_w = np.array([fwd[1], -fwd[0], 0.0])             # fwd rotated -90 deg about z
        o = obj_R.T @ right_w * (0.5 * hand_spacing)           # half-offset, object frame (in face plane)

        axis = int(np.argmax(np.abs(n)))
        p_c = p.copy()
        for a in range(3):
            if a != axis:
                lim = max(self.half[a] - self.face_margin, 0.0) - abs(o[a])
                if lim < 0:
                    raise ValueError(f"hand_spacing {hand_spacing} m does not fit on the face")
                p_c[a] = np.clip(p_c[a], -lim, lim)
        centered = np.allclose(p_c, p)
        pts = np.stack([p_c + o, p_c - o])                     # [right, left]
        return pts, np.stack([n, n]), centered


class HeuristicIntentionPlanner:
    '''
    Placeholder high level (stands in for the RL policy):
      - contacts: both hands on the pushable face opposite the push direction, symmetric
        about the point where the push line through the center meets the face
      - subgoal: the target, capped at max_step along the push direction
      - translate first; the orientation term is only switched on near the target
    '''
    def __init__(self, geometry: BoxGeometry, push_height=0.6, hand_spacing=0.4, max_step=0.3, ori_radius=0.1):
        self.geom = geometry
        self.push_height = push_height  # m above the ground
        self.hand_spacing = hand_spacing  # m between the two hand contacts
        self.max_step = max_step        # m per intention
        self.ori_radius = ori_radius    # m: enable the orientation subgoal within this distance

    def __call__(self, obj_pos, obj_R, target_pos, target_R) -> ContactIntention:
        obj_pos = np.asarray(obj_pos, dtype=float)
        target_pos = np.asarray(target_pos, dtype=float)
        delta = (target_pos - obj_pos)[:2]
        dist = np.linalg.norm(delta)

        if dist < self.ori_radius:
            # close enough: ask for the target orientation. Rotating needs off-center
            # contacts, which this heuristic does not choose yet (the RL policy's job).
            x_obj = obj_R[:2, 0]
            d_world = delta / dist if dist > 1e-6 else x_obj / np.linalg.norm(x_obj)
            subgoal_pos, w_ori = target_pos, 1.0
        else:
            d_world = delta / dist
            subgoal_pos = obj_pos.copy()
            subgoal_pos[:2] += d_world * min(self.max_step, dist)
            subgoal_pos[2] = target_pos[2]
            w_ori = 0.0

        pts, nrm, self.last_centered = self.geom.hand_push_points(
            d_world, obj_pos, obj_R, self.push_height, self.hand_spacing)
        return ContactIntention(
            contact_points=pts, contact_normals=nrm,
            subgoal_pos=subgoal_pos, subgoal_R=target_R,
            keypoint_ids=np.array([self.geom.nearest_keypoint(p) for p in pts]),
            w_pos=1.0, w_ori=w_ori,
        )
