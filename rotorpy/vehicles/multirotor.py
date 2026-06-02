from typing import List
import numpy as np
from numpy.linalg import inv, norm
import scipy.integrate
from scipy.spatial.transform import Rotation
# quad_params import left to caller — pass your trirotor_params dict directly.

from scipy.spatial.transform import Rotation as R

# imports for Batched Dynamics
try:
    import torch
    from torchdiffeq import odeint
    import roma
except ImportError:
    pass

import time

"""
Multirotor models — adapted for arbitrary rotor count, including trirotor (3 rotors).

Key changes vs. the original quadrotor-only version
----------------------------------------------------
1. Control allocation (Multirotor.__init__ and BatchedMultirotorParams.__init__)
   The original code called np.linalg.inv / torch.linalg.inv on f_to_TM, which
   requires a *square* matrix.  f_to_TM is (4 × num_rotors):
       • 4 rotors → 4×4, invertible   ✓
       • 3 rotors → 4×3, non-square   ✗  (raises LinAlgError)
   Fix: use np.linalg.pinv / torch.linalg.pinv (Moore-Penrose pseudoinverse).
   This gives the minimum-norm least-squares motor forces for any requested
   wrench, regardless of num_rotors.  For num_rotors == 4 and a full-rank
   square f_to_TM the pseudoinverse equals the regular inverse, so there is
   no regression for quadrotors.

2. State vector size (_pack_state / _unpack_state, both classes)
   The original code hardcoded size 20 (= 16 + 4 rotors).  Replaced with
   16 + num_rotors so it works for any rotor count.

3. Default initial_state rotor_speeds (Multirotor.__init__)
   Changed from np.array([1788.53]*4) to a num_rotors-length array of zeros.
   A caller should pass the correct hover speed for their vehicle.

4. BatchedMultirotor._unpack_state
   The 'rotor_speeds' slice was hardcoded to 4 elements; now uses num_rotors
   from the batched_params object.

All other physics (wrench computation, integrators, control abstractions) are
unchanged and generalise naturally through self.num_rotors / params.num_rotors.
"""


def quat_dot(quat, omega):
    """
    Parameters:
        quat, [i,j,k,w]
        omega, angular velocity of body in body axes

    Returns
        quat_dot, [i,j,k,w]
    """
    # Adapted from "Quaternions And Dynamics" by Basile Graf.
    (q0, q1, q2, q3) = (quat[0], quat[1], quat[2], quat[3])
    G = np.array([[ q3,  q2, -q1, -q0],
                  [-q2,  q3,  q0, -q1],
                  [ q1, -q0,  q3, -q2]])
    quat_dot = 0.5 * G.T @ omega
    # Rely on post-step renormalisation instead of a penalty term.
    return quat_dot


def quat_dot_torch(quat, omega):
    """
    Parameters:
        quat, (...,[i,j,k,w])
        omega, angular velocity of body in body axes: (...,3)

    Returns
        quat_dot, (...,[i,j,k,w])
    """
    b = quat.shape[0]
    # Adapted from "Quaternions And Dynamics" by Basile Graf.
    (q0, q1, q2, q3) = (quat[...,0], quat[...,1], quat[...,2], quat[...,3])
    G = torch.stack([q3, q2, -q1, -q0,
                     -q2, q3, q0, -q1,
                     q1, -q0, q3, -q2], dim=1).view((b, 3, 4))

    quat_dot = 0.5 * torch.transpose(G, 1, 2) @ omega.unsqueeze(-1)
    # Augment to maintain unit quaternion.
    quat_err = torch.sum(quat**2, dim=-1) - 1
    quat_err_grad = 2 * quat
    quat_dot = quat_dot.squeeze(-1) - quat_err.unsqueeze(-1) * quat_err_grad
    return quat_dot


class Multirotor(object):
    """
    Multirotor forward dynamics model.

    states: [position, velocity, attitude, body rates, wind, rotor speeds]

    Parameters:
        quad_params: a dictionary containing relevant physical parameters for
            the multirotor.
        initial_state: the initial state of the vehicle.
        control_abstraction: the appropriate control abstraction used by the
            controller, options are:
                'cmd_motor_speeds'  – controller commands motor speeds directly.
                'cmd_motor_thrusts' – controller commands forces for each rotor.
                'cmd_ctbr'          – collective thrust + body rates.
                'cmd_ctbm'          – collective thrust + body moments.
                'cmd_ctatt'         – collective thrust + attitude quaternion.
                'cmd_vel'           – velocity vector in the world frame.
                'cmd_acc'           – mass-normalised thrust vector (world frame).
        aero: bool, whether aerodynamic drag forces are computed.
        enable_ground: bool, whether ground contact is modelled.
        integrator_kwargs: dict passed to scipy.integrate.solve_ivp.
    """

    def __init__(self, quad_params,
                 initial_state=None,
                 control_abstraction='cmd_motor_speeds',
                 aero=True,
                 enable_ground=False,
                 integrator_kwargs=None):
        """
        Initialise multirotor physical parameters.
        """

        # Inertial parameters
        self.mass = quad_params['mass']   # kg
        self.Ixx  = quad_params['Ixx']   # kg·m²
        self.Iyy  = quad_params['Iyy']   # kg·m²
        self.Izz  = quad_params['Izz']   # kg·m²
        self.Ixy  = quad_params['Ixy']   # kg·m²
        self.Ixz  = quad_params['Ixz']   # kg·m²
        self.Iyz  = quad_params['Iyz']   # kg·m²

        # Frame parameters
        self.c_Dx = quad_params.get('c_Dx', 0.0)
        self.c_Dy = quad_params.get('c_Dy', 0.0)
        self.c_Dz = quad_params.get('c_Dz', 0.0)

        self.num_rotors  = quad_params['num_rotors']
        self.rotor_pos   = quad_params['rotor_pos']
        self.rotor_dir   = quad_params['rotor_directions']

        self.extract_geometry()

        # Rotor parameters
        self.rotor_speed_min = quad_params['rotor_speed_min']   # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max']   # rad/s

        self.k_eta   = quad_params['k_eta']
        self.k_m     = quad_params['k_m']
        self.k_d     = quad_params.get('k_d',    0.0)
        self.k_z     = quad_params.get('k_z',    0.0)
        self.k_h     = quad_params.get('k_h',    0.0)
        self.k_flap  = quad_params.get('k_flap', 0.0)

        # Motor parameters
        self.tau_m       = quad_params['tau_m']
        self.motor_noise = quad_params.get('motor_noise_std', 0)

        # Low-level controller gains
        self.k_w    = quad_params.get('k_w',    1)
        self.k_v    = quad_params.get('k_v',    10)
        self.kp_att = quad_params.get('kp_att', 3000.0)
        self.kd_att = quad_params.get('kd_att', 360.0)

        # Derived constants
        self.inertia = np.array([[self.Ixx, self.Ixy, self.Ixz],
                                 [self.Ixy, self.Iyy, self.Iyz],
                                 [self.Ixz, self.Iyz, self.Izz]])
        self.rotor_drag_matrix = np.array([[self.k_d, 0,        0      ],
                                           [0,        self.k_d, 0      ],
                                           [0,        0,        self.k_z]])
        self.drag_matrix = np.array([[self.c_Dx, 0,        0      ],
                                     [0,        self.c_Dy, 0      ],
                                     [0,        0,        self.c_Dz]])
        self.g = 9.81   # m/s²
        self._enable_ground = enable_ground
        self.ground_friction_beta = float(quad_params.get('ground_friction_beta', 0.3))
        self.ground_friction_beta = max(0.1, min(0.5, self.ground_friction_beta))

        self.inv_inertia = inv(self.inertia)
        self.weight = np.array([0, 0, -self.mass * self.g])

        # ------------------------------------------------------------------
        # Control allocation
        # f_to_TM maps individual rotor thrusts → [total_thrust, Mx, My, Mz].
        # Shape: (4, num_rotors).
        #
        # CHANGE: for num_rotors != 4 the matrix is non-square, so we use
        # np.linalg.pinv (Moore-Penrose pseudoinverse) instead of np.linalg.inv.
        # For a square full-rank matrix pinv == inv, so quadrotors are unaffected.
        # ------------------------------------------------------------------
        k = self.k_m / self.k_eta  # torque-to-thrust ratio

        self.f_to_TM = np.vstack((
            np.ones((1, self.num_rotors)),
            np.hstack([np.cross(self.rotor_pos[key], np.array([0, 0, 1]))
                         .reshape(-1, 1)[0:2]
                       for key in self.rotor_pos]),
            (k * self.rotor_dir).reshape(1, -1)
        ))
        # CHANGE: pinv instead of inv — supports non-square (trirotor) matrices.
        self.TM_to_f = np.linalg.pinv(self.f_to_TM)

        # Default initial state — num_rotors-aware
        if initial_state is None:
            initial_state = {
                'x':            np.array([0, 0, 0]),
                'v':            np.zeros(3),
                'q':            np.array([0, 0, 0, 1]),   # [i,j,k,w]
                'w':            np.zeros(3),
                'wind':         np.array([0, 0, 0]),
                # CHANGE: was hardcoded to 4 speeds; now matches num_rotors.
                'rotor_speeds': np.zeros(self.num_rotors),
            }
        self.initial_state = initial_state

        self.control_abstraction = control_abstraction
        self.aero = aero

        if integrator_kwargs is None:
            self.integrator_kwargs = {'method': 'RK45'}
        else:
            self.integrator_kwargs = integrator_kwargs

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def extract_geometry(self):
        """
        Build self.rotor_geometry as an (n, 3) array of rotor positions.
        """
        self.rotor_geometry = np.array([]).reshape(0, 3)
        for rotor in self.rotor_pos:
            self.rotor_geometry = np.vstack([self.rotor_geometry,
                                              self.rotor_pos[rotor]])

    # ------------------------------------------------------------------
    # State derivative helpers
    # ------------------------------------------------------------------

    def statedot(self, state, control, t_step):
        """
        Return the state derivative (vdot, wdot) at the current state without
        integrating, useful for logging or analysis.
        """
        cmd_rotor_speeds = self.get_cmd_motor_speeds(state, control)
        cmd_rotor_speeds = np.clip(cmd_rotor_speeds,
                                   self.rotor_speed_min, self.rotor_speed_max)

        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, cmd_rotor_speeds)

        s = Multirotor._pack_state(state, self.num_rotors)
        s_dot = s_dot_fn(0, s)
        return {'vdot': s_dot[3:6], 'wdot': s_dot[10:13]}

    def step(self, state, control, t_step):
        """
        Integrate dynamics forward from state given constant control for t_step.
        """
        cmd_rotor_speeds = self.get_cmd_motor_speeds(state, control)
        cmd_rotor_speeds = np.clip(cmd_rotor_speeds,
                                   self.rotor_speed_min, self.rotor_speed_max)

        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, cmd_rotor_speeds)

        s = Multirotor._pack_state(state, self.num_rotors)

        sol = scipy.integrate.solve_ivp(
            s_dot_fn, (0.0, t_step), s, **self.integrator_kwargs
        )
        s = sol['y'][:, -1]

        state = Multirotor._unpack_state(s, self.num_rotors)
        state['q'] = state['q'] / norm(state['q'])

        if self._enable_ground and self._on_ground(state):
            state = self._handle_vehicle_on_ground(state)

        state['rotor_speeds'] += np.random.normal(
            scale=np.abs(self.motor_noise), size=(self.num_rotors,))
        state['rotor_speeds'] = np.clip(state['rotor_speeds'],
                                         self.rotor_speed_min, self.rotor_speed_max)
        return state

    def _s_dot_fn(self, t, s, cmd_rotor_speeds):
        """
        Autonomous ODE: compute state derivative given fixed motor commands.
        """
        state = Multirotor._unpack_state(s, self.num_rotors)

        rotor_speeds      = state['rotor_speeds']
        inertial_velocity = state['v']
        wind_velocity     = state['wind']

        Rot = Rotation.from_quat(state['q']).as_matrix()

        rotor_accel = (1 / self.tau_m) * (cmd_rotor_speeds - rotor_speeds)
        x_dot = state['v']
        q_dot = quat_dot(state['q'], state['w'])

        body_airspeed_vector = Rot.T @ (inertial_velocity - wind_velocity)

        (FtotB, MtotB) = self.compute_body_wrench(
            state['w'], rotor_speeds, body_airspeed_vector)

        Ftot = Rot @ FtotB

        if self._enable_ground and self._on_ground(state):
            total_force = self.weight + Ftot
            if total_force[2] < 0:
                Ftot += np.array([0, 0, -total_force[2]])

        v_dot = (self.weight + Ftot) / self.mass

        w     = state['w']
        w_hat = Multirotor.hat_map(w)
        w_dot = self.inv_inertia @ (MtotB - w_hat @ (self.inertia @ w))

        wind_dot = np.zeros(3)

        # CHANGE: vector size is 16 + num_rotors (was hardcoded for 4 rotors).
        s_dot = np.zeros(16 + self.num_rotors)
        s_dot[0:3]   = x_dot
        s_dot[3:6]   = v_dot
        s_dot[6:10]  = q_dot
        s_dot[10:13] = w_dot
        s_dot[13:16] = wind_dot
        s_dot[16:]   = rotor_accel

        return s_dot

    def compute_body_wrench(self, body_rates, rotor_speeds, body_airspeed_vector):
        """
        Compute the wrench (force + moment) acting on the rigid body in the
        body frame, given current rotor speeds and body-frame airspeed.
        """
        local_airspeeds = (body_airspeed_vector[:, np.newaxis]
                           + Multirotor.hat_map(body_rates) @ self.rotor_geometry.T)

        T = np.array([0, 0, self.k_eta])[:, np.newaxis] * rotor_speeds**2

        if self.aero:
            D = (-Multirotor._norm(body_airspeed_vector)
                 * self.drag_matrix @ body_airspeed_vector)
            H = -rotor_speeds * (self.rotor_drag_matrix @ local_airspeeds)
            M_flap = (-self.k_flap * rotor_speeds
                      * ((Multirotor.hat_map(local_airspeeds.T)
                          .transpose(2, 0, 1)) @ np.array([0, 0, 1])).T)
            T += (np.array([0, 0, self.k_h])[:, np.newaxis]
                  * (local_airspeeds[0, :]**2 + local_airspeeds[1, :]**2))
        else:
            D      = np.zeros(3)
            H      = np.zeros((3, self.num_rotors))
            M_flap = np.zeros((3, self.num_rotors))

        M_force = -np.einsum('ijk, ik->j', Multirotor.hat_map(self.rotor_geometry), T + H)
        M_yaw   = self.rotor_dir * (np.array([0, 0, self.k_m])[:, np.newaxis] * rotor_speeds**2)

        FtotB = np.sum(T + H, axis=1) + D
        MtotB = M_force + np.sum(M_yaw + M_flap, axis=1)

        return FtotB, MtotB

    # ------------------------------------------------------------------
    # Control abstractions
    # ------------------------------------------------------------------

    def get_cmd_motor_speeds(self, state, control):
        """
        Convert the high-level control command into commanded motor speeds.
        """
        if self.control_abstraction == 'cmd_motor_speeds':
            return control['cmd_motor_speeds']

        elif self.control_abstraction == 'cmd_motor_thrusts':
            cmd_motor_speeds = control['cmd_motor_thrusts'] / self.k_eta
            return np.sign(cmd_motor_speeds) * np.sqrt(np.abs(cmd_motor_speeds))

        elif self.control_abstraction == 'cmd_ctbm':
            cmd_thrust = control['cmd_thrust']
            cmd_moment = control['cmd_moment']

        elif self.control_abstraction == 'cmd_ctbr':
            cmd_thrust = control['cmd_thrust']
            w_err      = state['w'] - control['cmd_w']
            wdot_cmd   = -self.k_w * w_err
            cmd_moment = self.inertia @ wdot_cmd

        elif self.control_abstraction == 'cmd_vel':
            v_err  = state['v'] - control['cmd_v']
            a_cmd  = -self.k_v * v_err
            F_des  = self.mass * (a_cmd + np.array([0, 0, self.g]))

            Rot    = Rotation.from_quat(state['q']).as_matrix()
            b3     = Rot @ np.array([0, 0, 1])
            cmd_thrust = np.dot(F_des, b3)

            b3_des = F_des / np.linalg.norm(F_des)
            c1_des = np.array([1, 0, 0])
            b2_des = np.cross(b3_des, c1_des) / np.linalg.norm(np.cross(b3_des, c1_des))
            b1_des = np.cross(b2_des, b3_des)
            R_des  = np.stack([b1_des, b2_des, b3_des]).T

            S_err  = 0.5 * (R_des.T @ Rot - Rot.T @ R_des)
            att_err = np.array([-S_err[1, 2], S_err[0, 2], -S_err[0, 1]])
            cmd_moment = (self.inertia @ (-self.kp_att * att_err - self.kd_att * state['w'])
                          + np.cross(state['w'], self.inertia @ state['w']))

        elif self.control_abstraction == 'cmd_ctatt':
            cmd_thrust = control['cmd_thrust']

            Rot   = Rotation.from_quat(state['q']).as_matrix()
            R_des = Rotation.from_quat(control['cmd_q']).as_matrix()

            S_err   = 0.5 * (R_des.T @ Rot - Rot.T @ R_des)
            att_err = np.array([-S_err[1, 2], S_err[0, 2], -S_err[0, 1]])
            cmd_moment = (self.inertia @ (-self.kp_att * att_err - self.kd_att * state['w'])
                          + np.cross(state['w'], self.inertia @ state['w']))

        elif self.control_abstraction == 'cmd_acc':
            F_des = control['cmd_acc'] * self.mass

            Rot    = Rotation.from_quat(state['q']).as_matrix()
            b3     = Rot @ np.array([0, 0, 1])
            cmd_thrust = np.dot(F_des, b3)

            b3_des = F_des / np.linalg.norm(F_des)
            c1_des = np.array([1, 0, 0])
            b2_des = np.cross(b3_des, c1_des) / np.linalg.norm(np.cross(b3_des, c1_des))
            b1_des = np.cross(b2_des, b3_des)
            R_des  = np.stack([b1_des, b2_des, b3_des]).T

            S_err   = 0.5 * (R_des.T @ Rot - Rot.T @ R_des)
            att_err = np.array([-S_err[1, 2], S_err[0, 2], -S_err[0, 1]])
            cmd_moment = (self.inertia @ (-self.kp_att * att_err - self.kd_att * state['w'])
                          + np.cross(state['w'], self.inertia @ state['w']))

        else:
            raise ValueError(
                "Invalid control abstraction. Options: cmd_motor_speeds, "
                "cmd_motor_thrusts, cmd_ctbm, cmd_ctbr, cmd_ctatt, cmd_vel, cmd_acc")

        TM = np.concatenate(([cmd_thrust], cmd_moment))
        cmd_motor_forces = self.TM_to_f @ TM
        cmd_motor_speeds = cmd_motor_forces / self.k_eta
        return np.sign(cmd_motor_speeds) * np.sqrt(np.abs(cmd_motor_speeds))

    # ------------------------------------------------------------------
    # Ground handling
    # ------------------------------------------------------------------

    def _on_ground(self, state):
        return state['x'][2] <= 0.001

    def _handle_vehicle_on_ground(self, state):
        state['x'][2]   = 0.0
        if state['v'][2] < 0.0:
            state['v'][2] = 0.0
        beta = self.ground_friction_beta
        state['v'][0:2] = (1.0 - beta) * state['v'][0:2]
        state['w']      = np.zeros(3)
        state['q']      = self.flatten_attitude(state['q'])
        return state

    # ------------------------------------------------------------------
    # Class / static methods
    # ------------------------------------------------------------------

    @classmethod
    def rotate_k(cls, q):
        return np.array([  2*(q[0]*q[2]+q[1]*q[3]),
                           2*(q[1]*q[2]-q[0]*q[3]),
                         1-2*(q[0]**2  +q[1]**2)  ])

    @classmethod
    def hat_map(cls, s):
        if len(s.shape) > 1:
            return np.array([[ np.zeros(s.shape[0]), -s[:, 2],  s[:, 1]],
                             [ s[:, 2],  np.zeros(s.shape[0]), -s[:, 0]],
                             [-s[:, 1],  s[:, 0],  np.zeros(s.shape[0])]])
        else:
            return np.array([[    0, -s[2],  s[1]],
                             [ s[2],     0, -s[0]],
                             [-s[1],  s[0],     0]])

    @classmethod
    def _pack_state(cls, state, num_rotors):
        """
        Convert a state dict to the flat internal vector.

        CHANGE: accepts num_rotors explicitly; size is 16 + num_rotors
        (was hardcoded to 20 = 16 + 4 rotors).
        """
        s = np.zeros(16 + num_rotors)
        s[0:3]   = state['x']
        s[3:6]   = state['v']
        s[6:10]  = state['q']
        s[10:13] = state['w']
        s[13:16] = state['wind']
        s[16:]   = state['rotor_speeds']
        return s

    @classmethod
    def _norm(cls, v):
        return (v[0]**2 + v[1]**2 + v[2]**2)**0.5

    @classmethod
    def _unpack_state(cls, s, num_rotors):
        """
        Convert the flat internal vector to a state dict.

        CHANGE: accepts num_rotors explicitly so the rotor_speeds slice is
        correct for any rotor count (was implicitly assuming 4 via s[16:20]).
        """
        return {
            'x':            s[0:3],
            'v':            s[3:6],
            'q':            s[6:10],
            'w':            s[10:13],
            'wind':         s[13:16],
            'rotor_speeds': s[16:16 + num_rotors],
        }

    @staticmethod
    def flatten_attitude(quaternion: List[float]) -> List[float]:
        """Set roll and pitch to 0, keeping yaw unchanged."""
        _, _, heading = R.from_quat(quaternion).as_euler('XYZ', degrees=False)
        return R.from_euler('Z', heading, degrees=False).as_quat()


# ===========================================================================
# Batched (PyTorch) implementation
# ===========================================================================

class BatchedMultirotorParams:
    """
    Parameter container for a batch of multirotors.

    Changes vs. the quadrotor-only original:
        • f_to_TM  is (num_drones, 4, num_rotors) — may be non-square.
        • TM_to_f  uses torch.linalg.pinv instead of torch.linalg.inv.
    """

    def __init__(self, multirotor_params_list, num_drones, device):
        assert len(multirotor_params_list) == num_drones
        self.num_drones = num_drones
        self.device = device

        self.mass = (torch.tensor([p['mass'] for p in multirotor_params_list])
                     .unsqueeze(-1).to(device))

        self.num_rotors = multirotor_params_list[0]['num_rotors']
        for p in multirotor_params_list:
            assert p['num_rotors'] == self.num_rotors, \
                "All drones in a batch must have the same num_rotors."

        self.rotor_pos     = [p['rotor_pos'] for p in multirotor_params_list]
        self.rotor_dir_np  = np.array([p['rotor_directions'] for p in multirotor_params_list])

        self.extract_geometry()

        self.rotor_speed_min = (torch.tensor([p['rotor_speed_min'] for p in multirotor_params_list])
                                .unsqueeze(-1).to(device))
        self.rotor_speed_max = (torch.tensor([p['rotor_speed_max'] for p in multirotor_params_list])
                                .unsqueeze(-1).to(device))

        self.k_eta  = np.array([p['k_eta']  for p in multirotor_params_list])
        self.k_m    = np.array([p['k_m']    for p in multirotor_params_list])
        self.k_flap = (torch.tensor([p['k_flap'] for p in multirotor_params_list])
                       .unsqueeze(-1).to(device))
        self.k_h    = (torch.tensor([p['k_h'] for p in multirotor_params_list])
                       .unsqueeze(-1).double().to(device))

        self.tau_m       = (torch.tensor([p['tau_m'] for p in multirotor_params_list], device=device)
                            .unsqueeze(-1))
        self.motor_noise = (torch.tensor([p['motor_noise_std'] for p in multirotor_params_list], device=device)
                            .unsqueeze(-1))

        self.inertia = torch.from_numpy(np.array([
            [[p['Ixx'], p['Ixy'], p['Ixz']],
             [p['Ixy'], p['Iyy'], p['Iyz']],
             [p['Ixz'], p['Iyz'], p['Izz']]]
            for p in multirotor_params_list
        ])).double().to(device)

        self.rotor_drag_matrix = torch.tensor([
            [[p['k_d'], 0,       0      ],
             [0,        p['k_d'], 0     ],
             [0,        0,        p['k_z']]]
            for p in multirotor_params_list
        ], device=device).double()

        self.drag_matrix = torch.tensor([
            [[p['c_Dx'], 0,        0      ],
             [0,         p['c_Dy'], 0     ],
             [0,         0,         p['c_Dz']]]
            for p in multirotor_params_list
        ], device=device).double()

        self.g = 9.81

        self.inv_inertia = torch.linalg.inv(self.inertia).double()
        self.weight = torch.zeros(num_drones, 3, device=device).double()
        self.weight[:, -1] = -self.mass.squeeze(-1) * self.g

        # Control allocation
        # CHANGE: f_to_TM is now (num_drones, 4, num_rotors) — non-square for
        # trirotors — and TM_to_f uses pinv instead of inv.
        k = self.k_m / self.k_eta  # shape: (num_drones,)

        self.f_to_TM = torch.stack([
            torch.from_numpy(np.vstack((
                np.ones((1, self.num_rotors)),
                np.hstack([
                    np.cross(self.rotor_pos[i][key], np.array([0, 0, 1]))
                    .reshape(-1, 1)[0:2]
                    for key in self.rotor_pos[i]
                ]),
                (k[i] * self.rotor_dir_np[i]).reshape(1, -1)
            ))).to(device)
            for i in range(num_drones)
        ])   # shape: (num_drones, 4, num_rotors)

        self.k_eta = torch.from_numpy(self.k_eta).unsqueeze(-1).to(device)
        self.k_m   = torch.from_numpy(self.k_m).unsqueeze(-1).to(device)
        self.rotor_dir = torch.from_numpy(self.rotor_dir_np).to(device)

        # CHANGE: pinv supports non-square f_to_TM (trirotor → 4×3).
        self.TM_to_f = torch.linalg.pinv(self.f_to_TM)

        # Low-level controller gains
        self.k_w    = (torch.tensor([p.get('k_w',    1)      for p in multirotor_params_list])
                       .unsqueeze(-1).to(device))
        self.k_v    = (torch.tensor([p.get('k_v',    10)     for p in multirotor_params_list])
                       .unsqueeze(-1).to(device))
        self.kp_att = (torch.tensor([p.get('kp_att', 3000.0) for p in multirotor_params_list])
                       .unsqueeze(-1).to(device))
        self.kd_att = (torch.tensor([p.get('kd_att', 360.0)  for p in multirotor_params_list])
                       .unsqueeze(-1).to(device))

    def update_mass(self, idx, mass):
        self.mass[idx]      = mass
        self.weight[idx,-1] = -mass * self.g

    def update_thrust_and_rotor_params(self, idx, k_eta=None, k_m=None, rotor_pos=None):
        if k_eta is not None:
            self.k_eta[idx] = k_eta
        if k_m is not None:
            self.k_m[idx] = k_m
        k_idx = self.k_m[idx] / self.k_eta[idx]
        if rotor_pos is not None:
            self.rotor_pos[idx] = dict(rotor_pos)
            rotor_geometry = np.array([]).reshape(0, 3)
            for rotor in rotor_pos:
                rotor_geometry = np.vstack([rotor_geometry, rotor_pos[rotor]])
            self.rotor_geometry[idx] = torch.from_numpy(rotor_geometry).double().to(self.device)
            self.rotor_geometry_hat_maps[idx] = (
                BatchedMultirotor.hat_map(torch.from_numpy(rotor_geometry.squeeze()))
                .double().to(self.device))

        self.f_to_TM[idx] = torch.from_numpy(np.vstack((
            np.ones((1, self.num_rotors)),
            np.hstack([
                np.cross(self.rotor_pos[idx][key], np.array([0, 0, 1]))
                .reshape(-1, 1)[0:2]
                for key in self.rotor_pos[idx]
            ]),
            (k_idx.cpu() * self.rotor_dir_np[idx]).reshape(1, -1)
        ))).to(self.device)
        # CHANGE: pinv here too.
        self.TM_to_f[idx] = torch.linalg.pinv(self.f_to_TM[idx])

    def update_inertia(self, idx, Ixx=None, Iyy=None, Izz=None):
        if Ixx is not None:
            self.inertia[idx][0, 0] = Ixx
        if Iyy is not None:
            self.inertia[idx][1, 1] = Iyy
        if Izz is not None:
            self.inertia[idx][2, 2] = Izz
        self.inv_inertia[idx] = torch.linalg.inv(self.inertia[idx])

    def update_drag(self, idx, c_Dx=None, c_Dy=None, c_Dz=None, k_d=None, k_z=None):
        if c_Dx is not None: self.drag_matrix[idx][0, 0] = c_Dx
        if c_Dy is not None: self.drag_matrix[idx][1, 1] = c_Dy
        if c_Dz is not None: self.drag_matrix[idx][2, 2] = c_Dz
        if k_d  is not None:
            self.rotor_drag_matrix[idx][0, 0] = k_d
            self.rotor_drag_matrix[idx][1, 1] = k_d
        if k_z  is not None:
            self.rotor_drag_matrix[idx][2, 2] = k_z

    def extract_geometry(self):
        geoms, geom_hat_maps = [], []
        for i in range(self.num_drones):
            rotor_geometry = np.array([]).reshape(0, 3)
            for rotor in self.rotor_pos[i]:
                rotor_geometry = np.vstack([rotor_geometry, self.rotor_pos[i][rotor]])
            geoms.append(rotor_geometry)
            geom_hat_maps.append(
                BatchedMultirotor.hat_map(torch.from_numpy(rotor_geometry.squeeze())).numpy())
        self.rotor_geometry = torch.from_numpy(np.array(geoms)).to(self.device)
        self.rotor_geometry_hat_maps = torch.from_numpy(np.array(geom_hat_maps)).to(self.device)


class BatchedMultirotor(object):
    """
    Batched multirotor forward dynamics (PyTorch).

    All changes vs. the original are in:
        • _pack_state   — state vector size is 16 + num_rotors (was 20).
        • _unpack_state — rotor_speeds slice uses num_rotors (was hardcoded 4).
        • BatchedMultirotorParams — TM_to_f uses pinv (see that class).
    Everything else is identical to the original.
    """

    def __init__(self, batched_params, num_drones, initial_states, device,
                 control_abstraction='cmd_motor_speeds',
                 aero=True,
                 integrator='dopri5'):
        assert initial_states['x'].device == device
        assert initial_states['x'].shape[0] == num_drones
        assert batched_params.device == device

        self.num_drones = num_drones
        self.device     = device
        self.params     = batched_params

        self.initial_states       = initial_states
        self.control_abstraction  = control_abstraction
        self.aero                 = aero

        assert integrator in ('dopri5', 'rk4')
        self.integrator = integrator

    def statedot(self, state, control, t_step, idxs):
        cmd_rotor_speeds = self.get_cmd_motor_speeds(state, control, idxs)
        cmd_rotor_speeds = torch.clip(cmd_rotor_speeds,
                                      self.params.rotor_speed_min[idxs],
                                      self.params.rotor_speed_max[idxs])

        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, cmd_rotor_speeds, idxs)

        s     = BatchedMultirotor._pack_state(state, self.num_drones, self.device,
                                              self.params.num_rotors)
        s_dot = s_dot_fn(0, s[idxs])

        v_dot = torch.zeros_like(state['v'])
        w_dot = torch.zeros_like(state['w'])
        v_dot[idxs] = s_dot[..., 3:6].double()
        w_dot[idxs] = s_dot[..., 10:13].double()

        return {'vdot': v_dot, 'wdot': w_dot}

    def step(self, state, control, t_step, idxs=None):
        if idxs is None:
            idxs = list(range(self.num_drones))

        cmd_rotor_speeds = self.get_cmd_motor_speeds(state, control, idxs)
        cmd_rotor_speeds = torch.clip(cmd_rotor_speeds,
                                      self.params.rotor_speed_min[idxs],
                                      self.params.rotor_speed_max[idxs])

        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, cmd_rotor_speeds, idxs)

        s = BatchedMultirotor._pack_state(state, self.num_drones, self.device,
                                          self.params.num_rotors)

        sol = odeint(s_dot_fn, s[idxs],
                     t=torch.tensor([0.0, t_step], device=self.device),
                     method=self.integrator)
        s = sol[-1, :]

        state = BatchedMultirotor._unpack_state(s, idxs, self.num_drones,
                                                 self.params.num_rotors)

        state['q'][idxs] = (state['q'][idxs]
                            / torch.norm(state['q'][idxs], dim=-1).unsqueeze(-1))

        state['rotor_speeds'][idxs] += torch.normal(
            mean=torch.zeros(self.params.num_rotors, device=self.device),
            std=(torch.ones(len(idxs), self.params.num_rotors, device=self.device)
                 * torch.abs(self.params.motor_noise[idxs])))
        state['rotor_speeds'][idxs] = torch.clip(
            state['rotor_speeds'][idxs],
            self.params.rotor_speed_min[idxs],
            self.params.rotor_speed_max[idxs])

        return state

    def _s_dot_fn(self, t, s, cmd_rotor_speeds, idxs):
        state = BatchedMultirotor._unpack_state(s, idxs, self.num_drones,
                                                 self.params.num_rotors)

        rotor_speeds      = state['rotor_speeds'][idxs]
        inertial_velocity = state['v'][idxs]
        wind_velocity     = state['wind'][idxs]

        Rot = roma.unitquat_to_rotmat(state['q'][idxs]).double()

        rotor_accel = (1 / self.params.tau_m[idxs]) * (cmd_rotor_speeds - rotor_speeds)
        x_dot = state['v'][idxs]
        q_dot = quat_dot_torch(state['q'][idxs], state['w'][idxs])

        body_airspeed_vector = (Rot.transpose(1, 2)
                                @ (inertial_velocity - wind_velocity).unsqueeze(-1).double())
        body_airspeed_vector = body_airspeed_vector.squeeze(-1)

        (FtotB, MtotB) = self.compute_body_wrench(
            state['w'][idxs], rotor_speeds, body_airspeed_vector, idxs)

        Ftot  = Rot @ FtotB.unsqueeze(-1)
        v_dot = ((self.params.weight[idxs] + Ftot.squeeze(-1))
                 / self.params.mass[idxs])

        w     = state['w'][idxs].double()
        w_hat = BatchedMultirotor.hat_map(w).permute(2, 0, 1)
        w_dot = self.params.inv_inertia[idxs] @ (
            MtotB - (w_hat.double()
                     @ (self.params.inertia[idxs] @ w.unsqueeze(-1))).squeeze(-1)
        ).unsqueeze(-1)

        wind_dot = torch.zeros((len(idxs), 3), device=self.device)

        # CHANGE: state vector is 16 + num_rotors (was hardcoded to 20).
        s_dot = torch.zeros((len(idxs), 16 + self.params.num_rotors), device=self.device)
        s_dot[:, 0:3]   = x_dot
        s_dot[:, 3:6]   = v_dot
        s_dot[:, 6:10]  = q_dot
        s_dot[:, 10:13] = w_dot.squeeze(-1)
        s_dot[:, 13:16] = wind_dot
        s_dot[:, 16:]   = rotor_accel

        return s_dot

    def compute_body_wrench(self, body_rates, rotor_speeds, body_airspeed_vector, idxs):
        num_drones = body_rates.shape[0]

        local_airspeeds = (body_airspeed_vector.unsqueeze(-1)
                           + (BatchedMultirotor.hat_map(body_rates).permute(2, 0, 1))
                           @ self.params.rotor_geometry[idxs].transpose(1, 2))

        T = torch.zeros(num_drones, 3, self.params.num_rotors, device=self.device)
        T[..., -1, :] = self.params.k_eta[idxs] * rotor_speeds**2

        if self.aero:
            tmp = self.params.drag_matrix[idxs] @ body_airspeed_vector.unsqueeze(-1)
            D = (-BatchedMultirotor._norm(body_airspeed_vector).unsqueeze(-1)
                 * tmp.squeeze())
            tmp = self.params.rotor_drag_matrix[idxs] @ local_airspeeds.double()
            H = -rotor_speeds.unsqueeze(1) * tmp

            M_flap = BatchedMultirotor.hat_map(
                local_airspeeds.transpose(1, 2).reshape(num_drones * self.params.num_rotors, 3))
            M_flap = (M_flap.permute(2, 0, 1)
                      .reshape(num_drones, self.params.num_rotors, 3, 3)
                      .double())
            M_flap = M_flap @ torch.tensor([0, 0, 1.0], device=self.device).double()
            M_flap = ((-self.params.k_flap[idxs] * rotor_speeds).unsqueeze(1)
                      * M_flap.transpose(-1, -2))

            lift = torch.zeros(num_drones, 3, 1, device=self.device).double()
            lift[:, 2, :] = self.params.k_h[idxs]
            lift = torch.bmm(
                lift,
                (local_airspeeds[:, 0, :]**2 + local_airspeeds[:, 1, :]**2).unsqueeze(1))
            T += lift
        else:
            D      = torch.zeros(num_drones, 3, device=self.device).double()
            H      = torch.zeros((num_drones, 3, self.params.num_rotors), device=self.device).double()
            M_flap = torch.zeros((num_drones, 3, self.params.num_rotors), device=self.device).double()

        M_force = -torch.einsum('bijk, bik->bj',
                                self.params.rotor_geometry_hat_maps[idxs], T + H)
        M_yaw = torch.zeros(num_drones, 3, self.params.num_rotors, device=self.device)
        M_yaw[..., -1, :] = (self.params.rotor_dir[idxs]
                              * self.params.k_m[idxs]
                              * rotor_speeds**2)

        FtotB = torch.sum(T + H, dim=2) + D
        MtotB = M_force + torch.sum(M_yaw + M_flap, dim=2)

        return FtotB, MtotB

    def get_cmd_motor_speeds(self, state, control, idxs):
        if self.control_abstraction == 'cmd_motor_speeds':
            return control['cmd_motor_speeds'][idxs]

        elif self.control_abstraction == 'cmd_motor_thrusts':
            cmd_motor_speeds = control['cmd_motor_thrusts'][idxs] / self.params.k_eta[idxs]
            return torch.sign(cmd_motor_speeds) * torch.sqrt(torch.abs(cmd_motor_speeds))

        elif self.control_abstraction == 'cmd_ctbm':
            cmd_thrust = control['cmd_thrust'][idxs]
            cmd_moment = control['cmd_moment'][idxs]

        elif self.control_abstraction == 'cmd_ctbr':
            cmd_thrust = control['cmd_thrust'][idxs]
            w_err      = state['w'][idxs] - control['cmd_w'][idxs]
            wdot_cmd   = -self.params.k_w[idxs] * w_err
            cmd_moment = self.params.inertia[idxs] @ wdot_cmd.unsqueeze(-1)

        elif self.control_abstraction == 'cmd_vel':
            v_err  = state['v'][idxs] - control['cmd_v'][idxs]
            a_cmd  = -self.params.k_v[idxs] * v_err
            F_des  = self.params.mass[idxs] * (a_cmd + np.array([0, 0, self.params.g]))

            Rot = roma.unitquat_to_rotmat(state['q'][idxs]).double()
            b3  = Rot @ torch.tensor([0.0, 0.0, 1.0], device=self.device).double()
            cmd_thrust = torch.sum(F_des * b3, dim=-1).double().unsqueeze(-1)

            b3_des = F_des / torch.norm(F_des, dim=-1, keepdim=True)
            c1_des = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).double()
            b2_des = (torch.cross(b3_des, c1_des, dim=-1)
                      / torch.norm(torch.cross(b3_des, c1_des, dim=-1), dim=-1, keepdim=True))
            b1_des = torch.cross(b2_des, b3_des, dim=-1)
            R_des  = torch.stack([b1_des, b2_des, b3_des], dim=-1)

            S_err   = 0.5 * (R_des.transpose(-1, -2) @ Rot - Rot.transpose(-1, -2) @ R_des)
            att_err = torch.stack([-S_err[:, 1, 2], S_err[:, 0, 2], -S_err[:, 0, 1]], dim=-1)

            Iw  = self.params.inertia[idxs] @ state['w'][idxs].unsqueeze(-1).double()
            tmp = -self.params.kp_att[idxs] * att_err - self.params.kd_att[idxs] * state['w']
            cmd_moment = ((self.params.inertia[idxs] @ tmp.unsqueeze(-1)).squeeze(-1)
                          + torch.cross(state['w'][idxs], Iw.squeeze(-1), dim=-1))

        elif self.control_abstraction == 'cmd_ctatt':
            cmd_thrust = control['cmd_thrust'][idxs]
            Rot   = roma.unitquat_to_rotmat(state['q'][idxs]).double()
            R_des = roma.unitquat_to_rotmat(control['cmd_q'][idxs]).double()
            S_err   = 0.5 * (R_des.transpose(-1, -2) @ Rot - Rot.transpose(-1, -2) @ R_des)
            att_err = torch.stack([-S_err[:, 1, 2], S_err[:, 0, 2], -S_err[:, 0, 1]], dim=-1)
            Iw  = self.params.inertia[idxs] @ state['w'][idxs].unsqueeze(-1).double()
            tmp = (-self.params.kp_att[idxs] * att_err
                   - self.params.kd_att[idxs] * state['w'][idxs])
            cmd_moment = ((self.params.inertia[idxs] @ tmp.unsqueeze(-1)).squeeze(-1)
                          + torch.cross(state['w'][idxs], Iw.squeeze(-1), dim=-1))

        elif self.control_abstraction == 'cmd_acc':
            F_des = control['cmd_acc'][idxs] * self.params.mass[idxs]
            Rot   = roma.unitquat_to_rotmat(state['q'][idxs]).double()
            b3    = Rot @ torch.tensor([0.0, 0.0, 1.0], device=self.device).double()
            cmd_thrust = torch.sum(F_des * b3, dim=-1).double().unsqueeze(-1)

            b3_des = F_des / torch.norm(F_des, dim=-1, keepdim=True)
            c1_des = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).double()
            b2_des = (torch.cross(b3_des, c1_des, dim=-1)
                      / torch.norm(torch.cross(b3_des, c1_des, dim=-1), dim=-1, keepdim=True))
            b1_des = torch.cross(b2_des, b3_des, dim=-1)
            R_des  = torch.stack([b1_des, b2_des, b3_des], dim=-1)

            S_err   = 0.5 * (R_des.transpose(-1, -2) @ Rot - Rot.transpose(-1, -2) @ R_des)
            att_err = torch.stack([-S_err[:, 1, 2], S_err[:, 0, 2], -S_err[:, 0, 1]], dim=-1)

            Iw  = self.params.inertia[idxs] @ state['w'][idxs].unsqueeze(-1).double()
            tmp = (-self.params.kp_att[idxs] * att_err
                   - self.params.kd_att[idxs] * state['w'])
            cmd_moment = ((self.params.inertia[idxs] @ tmp.unsqueeze(-1)).squeeze(-1)
                          + torch.cross(state['w'][idxs], Iw.squeeze(-1), dim=-1))

        else:
            raise ValueError(
                "Invalid control abstraction. Options: cmd_motor_speeds, "
                "cmd_motor_thrusts, cmd_ctbm, cmd_ctbr, cmd_ctatt, cmd_vel, cmd_acc")

        TM = torch.cat([cmd_thrust, cmd_moment.squeeze(-1)], dim=-1)
        # CHANGE: TM_to_f is now the pseudoinverse of the (4 × num_rotors) matrix.
        cmd_rotor_thrusts = (self.params.TM_to_f[idxs]
                             @ TM.unsqueeze(1).transpose(-1, -2)).squeeze(-1)
        cmd_motor_speeds = cmd_rotor_thrusts / self.params.k_eta[idxs]
        return torch.sign(cmd_motor_speeds) * torch.sqrt(torch.abs(cmd_motor_speeds))

    @classmethod
    def rotate_k(cls, q):
        return np.array([  2*(q[0]*q[2]+q[1]*q[3]),
                           2*(q[1]*q[2]-q[0]*q[3]),
                         1-2*(q[0]**2  +q[1]**2)  ])

    @classmethod
    def hat_map(cls, s):
        device = s.device
        if len(s.shape) > 1:
            s = s.unsqueeze(-1)
            hat = torch.cat([
                torch.zeros(s.shape[0], 1, device=device), -s[:, 2], s[:, 1],
                s[:, 2], torch.zeros(s.shape[0], 1, device=device), -s[:, 0],
                -s[:, 1], s[:, 0], torch.zeros(s.shape[0], 1, device=device)
            ], dim=0).view(3, 3, s.shape[0]).double()
            return hat
        else:
            return torch.tensor([[0, -s[2], s[1]],
                                  [s[2], 0, -s[0]],
                                  [-s[1], s[0], 0]], device=device)

    @classmethod
    def _pack_state(cls, state, num_drones, device, num_rotors):
        """
        CHANGE: accepts num_rotors; vector size is 16 + num_rotors.
        Was hardcoded to 20 (= 16 + 4 rotors).
        """
        s = torch.zeros(num_drones, 16 + num_rotors, device=device).double()
        s[..., 0:3]  = state['x']
        s[..., 3:6]  = state['v']
        s[..., 6:10] = state['q']
        s[..., 10:13]= state['w']
        s[..., 13:16]= state['wind']
        s[..., 16:]  = state['rotor_speeds']
        return s

    @classmethod
    def _norm(cls, v):
        return torch.linalg.norm(v, dim=-1)

    @classmethod
    def _unpack_state(cls, s, idxs, num_drones, num_rotors):
        """
        CHANGE: accepts num_rotors; rotor_speeds tensor is (num_drones, num_rotors).
        Was hardcoded to 4 via s[..., 16:20].
        """
        device = s.device
        state = {
            'x':            torch.full((num_drones, 3),          float('nan'), device=device).double(),
            'v':            torch.full((num_drones, 3),          float('nan'), device=device).double(),
            'q':            torch.full((num_drones, 4),          float('nan'), device=device).double(),
            'w':            torch.full((num_drones, 3),          float('nan'), device=device).double(),
            'wind':         torch.full((num_drones, 3),          float('nan'), device=device).double(),
            'rotor_speeds': torch.full((num_drones, num_rotors), float('nan'), device=device).double(),
        }
        state['q'][..., -1] = 1  # valid unit quaternion default
        state['x'][idxs]            = s[:, 0:3]
        state['v'][idxs]            = s[:, 3:6]
        state['q'][idxs]            = s[:, 6:10]
        state['w'][idxs]            = s[:, 10:13]
        state['wind'][idxs]         = s[:, 13:16]
        state['rotor_speeds'][idxs] = s[:, 16:16 + num_rotors]
        return state