import mujoco
import numpy as np


class DebugVisualizer:
    """
    Draws debug geoms into a passive viewer's user scene:
      - swing target:   a sphere at the current commanded swing-foot position
      - footstep plan:   a marker at the MPC's planned landing spot
      - foot frames:     RGB coordinate-axis triads at each foot

    Call clear() at the start of each frame, add_* to queue geoms, then the
    viewer.sync() in your loop renders them. user_scn persists between syncs,
    so clear() each tick to avoid accumulation.
    """

    def __init__(self, viewer):
        self.viewer = viewer
        self.scn = viewer.user_scn

    # ------------------------------------------------------------------ #
    def clear(self):
        self.scn.ngeom = 0

    def _next_geom(self):
        if self.scn.ngeom >= self.scn.maxgeom:
            return None
        g = self.scn.geoms[self.scn.ngeom]
        self.scn.ngeom += 1
        return g

    # ------------------------------------------------------------------ #
    def add_sphere(self, pos, radius=0.03, rgba=(1, 1, 0, 1)):
        g = self._next_geom()
        if g is None:
            return
        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius, 0, 0]),
            pos=np.asarray(pos, dtype=float),
            mat=np.eye(3).flatten(),
            rgba=np.asarray(rgba, dtype=float),
        )

    def add_swing_target(self, pos):
        self.add_sphere(pos, radius=0.06, rgba=(1.0, 0.9, 0.1, 1.0))  # yellow

    def add_footstep_plan(self, pos):
        self.add_sphere(pos, radius=0.04, rgba=(1.0, 0.2, 0.8, 0.7))   # magenta

    def add_com(self, pos):
        pos[2] = 0.0
        self.add_sphere(pos, radius=0.02, rgba=(1.0, 0, 0, 1.0)) # red

    def add_world_frame(self, length=0.25, width=0.008):
        """World frame at the origin: identity rotation, drawn slightly larger."""
        self.add_frame_axes(np.zeros(3), np.eye(3), length=length, width=width)

    def add_body_frame(self, mj_data, body_id, length=0.15, width=0.007):
        """Frame triad for any body, read from MuJoCo's xpos/xmat."""
        pos = mj_data.xpos[body_id]
        rot = mj_data.xmat[body_id].reshape(3, 3)
        self.add_frame_axes(pos, rot, length=length, width=width)



    # ------------------------------------------------------------------ #
    def add_frame_axes(self, pos, rot, length=0.12, width=0.006):
        """
        pos: (3,) world position of the frame origin
        rot: (3,3) rotation matrix (columns are the frame's x,y,z axes in world)
        Draws three capsules: red=x, green=y, blue=z.
        """
        pos = np.asarray(pos, dtype=float)
        rot = np.asarray(rot, dtype=float).reshape(3, 3)
        colors = [
            (1, 0, 0, 1),  # x red
            (0, 1, 0, 1),  # y green
            (0, 0, 1, 1),  # z blue
        ]
        for axis in range(3):
            g = self._next_geom()
            if g is None:
                return
            direction = rot[:, axis]
            end = pos + length * direction
            # connector capsule from pos to end
            mujoco.mjv_initGeom(
                g,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=np.zeros(3),
                pos=np.zeros(3),
                mat=np.eye(3).flatten(),
                rgba=np.asarray(colors[axis], dtype=float),
            )
            mujoco.mjv_connector(
                g,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                width,
                pos,
                end,
            )

    # ------------------------------------------------------------------ #
    def draw(self, mj_data, foot_body_ids, swing_target=None,
             footstep_plan=None, com=None, torso_body_id=None,
             draw_world=True):
        """
        Clear and draw foot frames + optional swing/plan markers, world frame,
        and torso frame.
        foot_body_ids: dict like {"right_foot": id, "left_foot": id}
        torso_body_id: int body id for the torso (optional)
        """
        self.clear()

        # Draw Coordinate Frames
        if draw_world:
            self.add_world_frame()

        for body_id in foot_body_ids.values():
            self.add_body_frame(mj_data, body_id, length=0.12, width=0.006)

        if torso_body_id is not None:
            self.add_body_frame(mj_data, torso_body_id, length=0.5, width=0.007)

        # Draw Positions
        if swing_target is not None:
            self.add_swing_target(swing_target)
        if footstep_plan is not None:
            self.add_footstep_plan(footstep_plan)
        if com is not None:
            self.add_com(com)