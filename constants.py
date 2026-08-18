import numpy as np

STEP_DURATION = 0.35  # seconds
XML_PATH = "xml_files/g1.xml"
FOOT_NAMES = ["right_ankle_roll_link", "left_ankle_roll_link"]
TORSO_NAME = "torso_link"
H = 15  # MPC horizon length

ACTUATOR_LIMITS = np.array([
    88, 139, 88, 139, 50, 50,      # left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    88, 139, 88, 139, 50, 50,      # right leg
    88, 50, 50,                    # waist: yaw, roll, pitch
    25, 25, 25, 25, 25, 5, 5,      # left arm: shoulder x3, elbow, wrist x3
    2.45, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4,  # left hand
    25, 25, 25, 25, 25, 5, 5,            # right arm
    2.45, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4,  # right hand
])

Z_H = 0.66 # m (slightly smaller than standing CoM)

MU = 0.3 # friction coefficient

# Kinematic Reachability
X_DIST = 0.4 # m
Y_DIST = 0.4 # m
STEP_WIDTH = 0.2 # m (from the stance foot) --> avoids stepping on toes

STEP_HEIGHT = 0.05 # m
FOOT_OFFSET = 0.03 # m (offset from frame origin to the ground, when foot is in contact)

MIN_FRAC = 0.6          # don't check for touchdown before this fraction of the nominal step
MAX_FRAC = 1.4          # force the switch here regardless, in case contact is never detected
HEIGHT_EPS = FOOT_OFFSET + 0.008