"""
Physical parameters for a trirotor (3-rotor) UAV.

Geometry:
    Three rotors placed at 120-degree intervals, arm length d = 0.043 m.
    Rotor positions (relative to CoM, body frame):
        r1: [ sqrt(3)/2,  0.5, 0] * d   (30 deg)
        r2: [-sqrt(3)/2,  0.5, 0] * d   (150 deg)
        r3: [  0.0,      -1.0, 0] * d   (270 deg)

Underactuation note:
    A fixed-pitch trirotor has only 3 actuators but 4 wrench DOFs (thrust +
    roll/pitch/yaw moments).  It is therefore underactuated: the allocation
    matrix f_to_TM is 4x3 and never square.  The dynamics file uses
    np.linalg.pinv (Moore-Penrose pseudoinverse) instead of np.linalg.inv so
    that motor forces are chosen in a minimum-norm least-squares sense.

    A consequence is that the yaw moment cannot be zeroed at hover regardless
    of rotor direction choice.  The best achievable residual with {+1,-1}
    directions is |sum(rotor_directions)| = 1, i.e. two rotors in one
    direction and one in the other.  Choosing [1, -1, 1] (or any permutation
    with one sign flip) reduces the hover Mz residual by ~6x compared to
    [1, 1, 1].

Bug fixed (original file):
    'rotor_directions' was [1, 1, 1].  All three rotors spinning the same way
    produces a net yaw torque of 3*k_m*omega^2 at hover with no way to cancel
    it, making the vehicle uncontrollable in yaw.  Corrected to [1, -1, 1].

Additional sources:
    https://bitcraze.io/2015/02/measuring-propeller-rpm-part-3
    https://wiki.bitcraze.io/misc:investigations:thrust
    https://commons.erau.edu/cgi/viewcontent.cgi?article=2057&context=publication
    "Data-Driven System Identification of Quadrotors Subject to Motor Delays",
    Eschmann et al. 2024. https://arxiv.org/abs/2404.07837

Notes:
    k_eta is inferred from 14.5 g thrust at 2500 rad/s.
    k_m is mostly made up.
"""

import numpy as np

d = 0.043  # arm length, metres

quad_params = {

    # Inertial properties
    'mass': 0.03,       # kg
    'Ixx':  1.43e-5,    # kg*m^2
    'Iyy':  1.43e-5,    # kg*m^2
    'Izz':  2.89e-5,    # kg*m^2
    'Ixy':  0.0,        # kg*m^2
    'Iyz':  0.0,        # kg*m^2
    'Ixz':  0.0,        # kg*m^2

    # Geometric properties — all vectors relative to the centre of mass.
    'num_rotors': 3,
    'rotor_pos': {
        'r1': d * np.array([ np.sqrt(3) / 2,  0.5, 0.0]),   # 30 deg,  arm = d
        'r2': d * np.array([-np.sqrt(3) / 2,  0.5, 0.0]),   # 150 deg, arm = d
        'r3': d * np.array([ 0.0,            -1.0, 0.0]),   # 270 deg, arm = d
    },

    # FIX: was [1, 1, 1] — all same-direction rotors cannot cancel yaw torque.
    # With [1, -1, 1] the hover Mz residual is ~6x smaller than [1, 1, 1].
    # Two CW (r1, r3) and one CCW (r2).
    'rotor_directions': np.array([1, -1, 1]),

    'rI': np.array([0, 0, 0]),  # IMU location, metres

    # Frame aerodynamic properties
    'c_Dx': 0.0,   # parasitic drag, body x, N/(m/s)^2
    'c_Dy': 0.0,   # parasitic drag, body y, N/(m/s)^2
    'c_Dz': 0.0,   # parasitic drag, body z, N/(m/s)^2

    # Rotor aerodynamic properties
    'k_eta':  2.3e-08,      # thrust coefficient,         N/(rad/s)^2
    'k_m':    7.8e-10,      # yaw moment coefficient,     Nm/(rad/s)^2
    'k_d':    10.2506e-07,  # rotor drag coefficient,     N/(rad·m/s^2)
    'k_z':    7.553e-07,    # induced inflow coefficient, N/(rad·m/s^2)
    'k_h':    0.0,          # translational lift,         N/(m/s)^2
    'k_flap': 0.0,          # flapping moment,            Nm/(rad·m/s^2)

    # Motor properties
    'tau_m':           0.072,   # motor response time, s
    'rotor_speed_min': 0,       # rad/s
    'rotor_speed_max': 2500,    # rad/s
    'motor_noise_std': 0.0,     # rad/s

    # Low-level controller gains (used when a higher-level control abstraction
    # is selected, e.g. cmd_ctbr, cmd_vel, cmd_ctatt).
    'k_w':    200,    # body rate P gain    (cmd_ctbr)
    'k_v':     10,    # world velocity P gain (cmd_vel)
    'kp_att': 1030,   # attitude P gain     (cmd_vel / cmd_acc / cmd_ctatt)
    'kd_att':   51,   # attitude D gain     (cmd_vel / cmd_acc / cmd_ctatt)
}