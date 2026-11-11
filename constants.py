STEP_DURATION = 0.30  # seconds
SCENE_PATH = "xml_files/scene.xml"
ROBOT_PATH = "xml_files/humanoid_3D.xml"  # was biped_3d_5dof_leg.xml
H = 4  # MPC horizon length
ACTUATOR_LIMIT = 50 # Nm  (unused for humanoid: per-joint limits come from the XML forcerange)
STEP_WIDTH = 0.2 # m (from the stance foot)
Z_H = 1.23 # m  (humanoid standing CoM is 1.31; biped used 1.20 vs 1.29 standing)
MU = 0.5 # friction coefficient

# Arm joint setpoints (degrees), held by a joint-space PD in the WBC.
# All zeros = arms straight out to the sides (T-pose). Ranges: shoulders +/-90,
# right elbow -120..0, left elbow 0..120. E.g. arms down: right roll 80, left roll -80.
ARM_SETPOINTS = {
    "right_shoulder_yaw": 0.0, "right_shoulder_roll": 0.0, "right_shoulder_pitch": 0.0, "right_elbow": 0.0,
    "left_shoulder_yaw":  0.0, "left_shoulder_roll":  0.0, "left_shoulder_pitch":  0.0, "left_elbow":  0.0,
}

MIN_FRAC = 0.6          # don't even check for touchdown before this fraction of the nominal step
MAX_FRAC = 1.4          # force the switch here regardless, in case contact is never detected
HEIGHT_EPS = 0.06       # just above final_swing_target[2] (0.05) so it triggers on genuine touchdown