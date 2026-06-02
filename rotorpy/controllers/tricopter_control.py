import numpy as np

try:
    import torch
    import roma
except ImportError:
    pass
from scipy.spatial.transform import Rotation


class SE3Control(object):
    """
    Multirotor trajectory tracking controller based on
    https://ieeexplore.ieee.org/document/5717652

    Works for any number of rotors. For an underactuated vehicle such as a
    trirotor (3 fixed-pitch rotors) the control allocation uses the
    Moore-Penrose pseudoinverse, giving minimum-norm motor forces in a
    least-squares sense.  See the dynamics file for a full discussion of
    trirotor underactuation.
    """

    def __init__(self, quad_params):
        """
        Parameters:
            quad_params, dict with keys specified in rotorpy/vehicles
        """

        # Inertial parameters
        self.mass = quad_params['mass']  # kg
        self.Ixx = quad_params['Ixx']  # kg*m^2
        self.Iyy = quad_params['Iyy']  # kg*m^2
        self.Izz = quad_params['Izz']  # kg*m^2
        self.Ixy = quad_params['Ixy']  # kg*m^2
        self.Ixz = quad_params['Ixz']  # kg*m^2
        self.Iyz = quad_params['Iyz']  # kg*m^2

        # Frame parameters
        self.c_Dx = quad_params['c_Dx']
        self.c_Dy = quad_params['c_Dy']
        self.c_Dz = quad_params['c_Dz']

        self.num_rotors = quad_params['num_rotors']
        self.rotor_pos = quad_params['rotor_pos']
        self.rotor_dir = quad_params['rotor_directions']

        # Rotor parameters
        self.rotor_speed_min = quad_params['rotor_speed_min']  # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max']  # rad/s

        self.k_eta = quad_params['k_eta']
        self.k_m = quad_params['k_m']
        self.k_d = quad_params['k_d']
        self.k_z = quad_params['k_z']
        self.k_flap = quad_params['k_flap']

        # Motor parameters
        self.tau_m = quad_params['tau_m']

        self.inertia = np.array([[self.Ixx, self.Ixy, self.Ixz],
                                 [self.Ixy, self.Iyy, self.Iyz],
                                 [self.Ixz, self.Iyz, self.Izz]])  # kg*m^2
        self.g = 9.81  # m/s^2

        # Control gains
        self.kp_pos = np.array([6.5, 6.5, 15])
        self.kd_pos = np.array([4.0, 4.0, 9])
        self.kp_att = 544
        self.kd_att = 46.64
        self.kp_vel = 0.1 * self.kp_pos

        # Control allocation: maps individual rotor forces → [thrust, Mx, My, Mz].
        # f_to_TM is (4, num_rotors); for num_rotors == 4 it is square; for
        # num_rotors == 3 (trirotor) it is (4, 3) and non-invertible.
        k = self.k_m / self.k_eta

        self.f_to_TM = np.vstack((
            np.ones((1, self.num_rotors)),
            np.hstack([np.cross(self.rotor_pos[key], np.array([0, 0, 1]))
                      .reshape(-1, 1)[0:2]
                       for key in self.rotor_pos]),
            (k * self.rotor_dir).reshape(1, -1)
        ))
        # CHANGE: pinv instead of inv.
        # np.linalg.inv requires a square matrix and raises LinAlgError for
        # a trirotor's (4 × 3) f_to_TM.  pinv (Moore-Penrose pseudoinverse)
        # works for any shape and equals inv when the matrix is square and
        # full-rank, so quadrotor behaviour is unchanged.
        self.TM_to_f = np.linalg.pinv(self.f_to_TM)

    def update(self, t, state, flat_output):
        """
        Compute control inputs from current state and desired flat outputs.

        Inputs:
            t,           present time in seconds
            state,       dict with keys: x, v, q [i,j,k,w], w
            flat_output, dict with keys: x, x_dot, x_ddot, x_dddot, x_ddddot,
                         yaw, yaw_dot

        Outputs:
            control_input dict with keys:
                cmd_motor_speeds  (num_rotors,) rad/s
                cmd_motor_thrusts (num_rotors,) N
                cmd_thrust        scalar N
                cmd_moment        (3,) N·m
                cmd_q             quaternion [i,j,k,w]
                cmd_w             body angular rates rad/s
                cmd_v             world-frame velocity m/s
                cmd_acc           mass-normalised thrust vector m/s²
        """
        # CHANGE: initialise to num_rotors elements, not a hardcoded 4.
        # The original np.zeros((4,)) caused a shape mismatch when TM_to_f
        # produced a (3,)-length result for a trirotor.
        cmd_motor_speeds = np.zeros(self.num_rotors)
        cmd_thrust = 0
        cmd_moment = np.zeros(3)
        cmd_q = np.zeros(4)  # quaternion — always 4 elements, unchanged

        def normalize(x):
            return x / np.linalg.norm(x)

        def vee_map(S):
            return np.array([-S[1, 2], S[0, 2], -S[0, 1]])

        # Desired force vector
        pos_err = state['x'] - flat_output['x']
        dpos_err = state['v'] - flat_output['x_dot']
        F_des = self.mass * (-self.kp_pos * pos_err
                             - self.kd_pos * dpos_err
                             + flat_output['x_ddot']
                             + np.array([0, 0, self.g]))

        # Desired thrust projected onto body z axis
        R = Rotation.from_quat(state['q']).as_matrix()
        b3 = R @ np.array([0, 0, 1])
        u1 = np.dot(F_des, b3)

        # Desired orientation
        b3_des = normalize(F_des)
        yaw_des = flat_output['yaw']
        c1_des = np.array([np.cos(yaw_des), np.sin(yaw_des), 0])
        b2_des = normalize(np.cross(b3_des, c1_des))
        b1_des = np.cross(b2_des, b3_des)
        R_des = np.stack([b1_des, b2_des, b3_des]).T

        # Orientation error
        S_err = 0.5 * (R_des.T @ R - R.T @ R_des)
        att_err = vee_map(S_err)

        # Angular velocity error
        w_des = np.array([0, 0, flat_output['yaw_dot']])
        w_err = state['w'] - w_des

        # Desired torque (N·m)
        u2 = (self.inertia @ (-self.kp_att * att_err - self.kd_att * w_err)
              + np.cross(state['w'], self.inertia @ state['w']))

        # Commanded body rates (for cmd_ctbr abstraction)
        cmd_w = -self.kp_att * att_err - self.kd_att * w_err

        # Motor speeds via control allocation.
        # TM is always the 4-element wrench vector [thrust, Mx, My, Mz].
        # TM_to_f is (num_rotors, 4), so cmd_rotor_thrusts is (num_rotors,).
        TM = np.array([u1, u2[0], u2[1], u2[2]])
        cmd_rotor_thrusts = self.TM_to_f @ TM
        cmd_motor_speeds = cmd_rotor_thrusts / self.k_eta
        cmd_motor_speeds = np.sign(cmd_motor_speeds) * np.sqrt(np.abs(cmd_motor_speeds))

        cmd_thrust = u1
        cmd_moment = u2
        cmd_q = Rotation.from_matrix(R_des).as_quat()
        cmd_v = -self.kp_vel * pos_err + flat_output['x_dot']
        cmd_acc = F_des / self.mass

        return {'cmd_motor_speeds': cmd_motor_speeds,
                'cmd_motor_thrusts': cmd_rotor_thrusts,
                'cmd_thrust': cmd_thrust,
                'cmd_moment': cmd_moment,
                'cmd_q': cmd_q,
                'cmd_w': cmd_w,
                'cmd_v': cmd_v,
                'cmd_acc': cmd_acc}


class BatchedSE3Control(object):
    """
    Batched SE3 controller for any number of rotors.

    Changes vs. the original:
        • _unpack_control receives num_rotors and uses it instead of the
          hardcoded 4 when allocating cmd_motor_speeds / cmd_motor_thrusts.
        • update() passes self.params.num_rotors through to _unpack_control.
    TM_to_f already comes from BatchedMultirotorParams (which now uses pinv),
    so no change is needed here for the allocation itself.
    """

    def __init__(self, batch_params, num_drones, device,
                 kp_pos=None, kd_pos=None, kp_att=None, kd_att=None):
        """
        Parameters:
            batch_params: BatchedMultirotorParams object
            num_drones:   int
            device:       torch.device
            kp_pos:       (num_drones, 3) or None
            kd_pos:       (num_drones, 3) or None
            kp_att:       (num_drones, 1) or None
            kd_att:       (num_drones, 1) or None
        """
        assert batch_params.device == device
        self.params = batch_params
        self.device = device

        if kp_pos is None:
            self.kp_pos = torch.tensor([6.5, 6.5, 15], device=device).repeat(num_drones, 1).double()
        else:
            self.kp_pos = kp_pos.to(device).double()

        if kd_pos is None:
            self.kd_pos = torch.tensor([4.0, 4.0, 9], device=device).repeat(num_drones, 1).double()
        else:
            self.kd_pos = kd_pos.to(device).double()

        if kp_att is None:
            self.kp_att = torch.tensor([544], device=device).repeat(num_drones, 1).double()
        else:
            self.kp_att = kp_att.to(device).double()
            if len(self.kp_att.shape) < 2:
                self.kp_att = self.kp_att.unsqueeze(-1)

        if kd_att is None:
            self.kd_att = torch.tensor([46.64], device=device).repeat(num_drones, 1).double()
        else:
            self.kd_att = kd_att.to(device).double()
            if len(self.kd_att.shape) < 2:
                self.kd_att = self.kd_att.unsqueeze(-1)

        self.kp_vel = 0.1 * self.kp_pos

    def normalize(self, x):
        return x / torch.norm(x, dim=-1, keepdim=True)

    def update(self, t, states, flat_outputs, idxs=None):
        """
        Compute a batch of control outputs for the drones at idxs.

        Parameters:
            states:       dict of double-precision torch tensors
            flat_outputs: dict of double-precision torch tensors
            idxs:         list of drone indices to update (default: all)

        Returns:
            control_inputs dict of torch tensors
        """
        if idxs is None:
            idxs = list(range(states['x'].shape[0]))

        pos_err = states['x'][idxs].double() - flat_outputs['x'][idxs].double()
        dpos_err = states['v'][idxs].double() - flat_outputs['x_dot'][idxs].double()

        F_des = self.params.mass[idxs] * (-self.kp_pos[idxs] * pos_err
                                          - self.kd_pos[idxs] * dpos_err
                                          + flat_outputs['x_ddot'][idxs].double()
                                          + torch.tensor([0, 0, self.params.g],
                                                         device=self.device))

        R = roma.unitquat_to_rotmat(states['q'][idxs]).double()
        b3 = R @ torch.tensor([0.0, 0.0, 1.0], device=self.device).double()
        u1 = torch.sum(F_des * b3, dim=-1).double()

        b3_des = self.normalize(F_des)
        yaw_des = flat_outputs['yaw'][idxs].double()
        c1_des = torch.stack([torch.cos(yaw_des), torch.sin(yaw_des),
                              torch.zeros_like(yaw_des)], dim=-1)
        b2_des = self.normalize(torch.cross(b3_des, c1_des, dim=-1))
        b1_des = torch.cross(b2_des, b3_des, dim=-1)
        R_des = torch.stack([b1_des, b2_des, b3_des], dim=-1)

        S_err = 0.5 * (R_des.transpose(-1, -2) @ R - R.transpose(-1, -2) @ R_des)
        att_err = torch.stack([-S_err[:, 1, 2], S_err[:, 0, 2], -S_err[:, 0, 1]], dim=-1)

        w_des = torch.stack([torch.zeros_like(yaw_des),
                             torch.zeros_like(yaw_des),
                             flat_outputs['yaw_dot'][idxs].double()], dim=-1).to(self.device)
        w_err = states['w'][idxs].double() - w_des

        Iw = self.params.inertia[idxs] @ states['w'][idxs].unsqueeze(-1).double()
        tmp = -self.kp_att[idxs] * att_err - self.kd_att[idxs] * w_err
        u2 = ((self.params.inertia[idxs] @ tmp.unsqueeze(-1)).squeeze(-1)
              + torch.cross(states['w'][idxs].double(), Iw.squeeze(-1), dim=-1))

        # TM is always (batch, 4): [thrust, Mx, My, Mz].
        # self.params.TM_to_f[idxs] is (batch, num_rotors, 4) via pinv,
        # so cmd_rotor_thrusts is (batch, num_rotors).
        TM = torch.cat([u1.unsqueeze(-1), u2], dim=-1)
        cmd_rotor_thrusts = (self.params.TM_to_f[idxs]
                             @ TM.unsqueeze(1).transpose(-1, -2)).squeeze(-1)
        cmd_motor_speeds = cmd_rotor_thrusts / self.params.k_eta[idxs]
        cmd_motor_speeds = torch.sign(cmd_motor_speeds) * torch.sqrt(torch.abs(cmd_motor_speeds))

        cmd_q = roma.rotmat_to_unitquat(R_des)
        cmd_v = -self.kp_vel[idxs] * pos_err + flat_outputs['x_dot'][idxs].double()

        # CHANGE: pass self.params.num_rotors so _unpack_control allocates
        # tensors of the correct width instead of the hardcoded 4.
        return BatchedSE3Control._unpack_control(
            cmd_motor_speeds,
            cmd_rotor_thrusts,
            u1.unsqueeze(-1),
            u2,
            cmd_q,
            -self.kp_att[idxs] * att_err - self.kd_att[idxs] * w_err,
            cmd_v,
            F_des / self.params.mass[idxs],
            idxs,
            states['x'].shape[0],
            self.params.num_rotors,  # CHANGE: new argument
        )

    @classmethod
    def _unpack_control(cls, cmd_motor_speeds, cmd_motor_thrusts,
                        u1, u2, cmd_q, cmd_w, cmd_v, cmd_acc,
                        idxs, num_drones, num_rotors):
        """
        Pack computed quantities into the control dict expected by the dynamics.

        CHANGE: accepts num_rotors and uses it for cmd_motor_speeds /
        cmd_motor_thrusts tensor widths.  Was hardcoded to 4, which caused a
        shape mismatch for any vehicle with num_rotors != 4.
        """
        device = cmd_motor_speeds.device
        ctrl = {
            # CHANGE: second dim is num_rotors, not 4.
            'cmd_motor_speeds': torch.zeros(num_drones, num_rotors, dtype=torch.double, device=device),
            'cmd_motor_thrusts': torch.zeros(num_drones, num_rotors, dtype=torch.double, device=device),
            'cmd_thrust': torch.zeros(num_drones, 1, dtype=torch.double, device=device),
            'cmd_moment': torch.zeros(num_drones, 3, dtype=torch.double, device=device),
            'cmd_q': torch.zeros(num_drones, 4, dtype=torch.double, device=device),
            'cmd_w': torch.zeros(num_drones, 3, dtype=torch.double, device=device),
            'cmd_v': torch.zeros(num_drones, 3, dtype=torch.double, device=device),
            'cmd_acc': torch.zeros(num_drones, 3, dtype=torch.double, device=device),
        }
        ctrl['cmd_motor_speeds'][idxs] = cmd_motor_speeds
        ctrl['cmd_motor_thrusts'][idxs] = cmd_motor_thrusts
        ctrl['cmd_thrust'][idxs] = u1
        ctrl['cmd_moment'][idxs] = u2
        ctrl['cmd_q'][idxs] = cmd_q
        ctrl['cmd_w'][idxs] = cmd_w
        ctrl['cmd_v'][idxs] = cmd_v
        ctrl['cmd_acc'][idxs] = cmd_acc
        return ctrl
