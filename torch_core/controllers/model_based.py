import torch
import torch.nn as nn
from torch_core.simulator.vessels.csad_torch import CSAD_6DOF
from torch_core.utils import six2threeDOF


class ModelController(nn.Module):
    """
    A simple model-based controller for the ship using the full dynamic model.
    
    Implements a PD control law with model-based compensation.
    """
    def __init__(self, dt):
        super().__init__()
        self.dt = dt

        # Load vessel dynamics (M and D matrices)
        self.simulator = CSAD_6DOF(dt)
        self.M_inv = six2threeDOF(self.simulator._Minv)
        self.M = six2threeDOF(self.simulator._M)
        self.D = six2threeDOF(self.simulator._D)
        

        # PD Gains (adjustable)
        self.Kp = torch.tensor([5.0, 10.0, 2.5], dtype=torch.float32)  # Proportional gains
        self.Kd = torch.tensor([2.0, 10.0, 1], dtype=torch.float32)  # Derivative gains

    def compute_control(self, state, eta_d, nu_d, eta_d_ddot):
        """
        Compute control input using PD control + model-based compensation.

        Parameters:
        - state (dict): Current state of the vessel
        - eta_d (Tensor): Desired position [x, y, psi]
        - nu_d (Tensor): Desired velocity [u, v, r]
        - eta_d_ddot (Tensor): Desired acceleration

        Returns:
        - tau (Tensor): Control forces [Fx, Fy, Mz]
        """
        eta = six2threeDOF(torch.tensor(state["eta"], dtype=torch.float32))  # Current position
        nu = torch.tensor(state["nu"], dtype=torch.float32)  # Current velocity

        # Compute tracking error
        error_eta = eta_d - eta  # Position error
        error_nu = nu_d - nu  # Velocity error

        tau = self.M@(eta_d_ddot + self.Kp * error_eta + self.Kd * error_nu) + self.D@nu  # PD control + model-based compensation

        return tau