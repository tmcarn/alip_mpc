STEP_DURATION = 0.30  # seconds
XML_PATH = "xml_files/biped_3d_5dof_leg.xml"
H = 4  # MPC horizon length
ACTUATOR_LIMIT = 50 # Nm
STEP_WIDTH = 0.2 # m (from the stance foot)
Z_H = 1.3 # m
MU = 0.6 # friction coefficient

MIN_FRAC = 0.6          # don't even check for touchdown before this fraction of the nominal step
MAX_FRAC = 1.4          # force the switch here regardless, in case contact is never detected
HEIGHT_EPS = 0.06       # just above final_swing_target[2] (0.05) so it triggers on genuine touchdown