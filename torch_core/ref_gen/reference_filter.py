#!/usr/bin/env python3
"""
reference_filter.py
TODO: Fix hardcoded cpu device, should be updated to use the device of the input tensors.

Differentiable third-order reference filter for guidance using PyTorch.
This module replicates the original ThrdOrderRefFilter functionality but uses torch.Tensors
for all computations, making it fully differentiable.

Author: Kristian Magnus Roen 
Date: 2025-02-17
"""

import torch
from torch_core.utils import Rz_torch


class ThrdOrderRefFilter(torch.nn.Module):
    """
    Differentiable third-order reference filter.
    
    Attributes:
      eta_d       : Desired vessel pose (tensor of shape (3,))
      eta_d_dot   : Desired vessel velocity (tensor of shape (3,))
      eta_d_ddot  : Desired vessel acceleration (tensor of shape (3,))
    """
    def __init__(self, dt, omega=[0.2, 0.2, 0.1], initial_eta=None):
        super(ThrdOrderRefFilter, self).__init__()
        self._dt = dt
        device = torch.device("cpu")
        if initial_eta is None:
            self.eta_d = torch.zeros(3, dtype=torch.float32, device=device)
        else:
            self.eta_d = torch.tensor(initial_eta, dtype=torch.float32, device=device)
        self.eta_d_dot = torch.zeros(3, dtype=torch.float32, device=device)
        self.eta_d_ddot = torch.zeros(3, dtype=torch.float32, device=device)
        self._eta_r = self.eta_d.clone()
        # Concatenate desired state into a state vector _x of shape (9,)
        self._x = torch.cat([self.eta_d, self.eta_d_dot, self.eta_d_ddot])
        # Define delta as 3x3 identity
        self._delta = torch.eye(3, dtype=torch.float32, device=device)
        # Create a diagonal omega matrix
        self._w = torch.diag(torch.tensor(omega, dtype=torch.float32, device=device))
        O3 = torch.zeros((3, 3), dtype=torch.float32, device=device)
        I3 = torch.eye(3, dtype=torch.float32, device=device)
        # Build the 9x9 block matrix Ad using torch.cat
        # Ad = [ [O3, I3, O3],
        #        [O3, O3, I3],
        #        [-w^3, -(2*delta + I3)w^2, -(2*delta + I3)w] ]
        w3 = self._w @ self._w @ self._w
        w2 = self._w @ self._w
        term = (2 * self._delta + I3)
        row1 = torch.cat([O3, I3, O3], dim=1)  # (3,9)
        row2 = torch.cat([O3, O3, I3], dim=1)  # (3,9)
        row3 = torch.cat([-w3, -term @ w2, -term @ self._w], dim=1)  # (3,9)
        self.Ad = torch.cat([row1, row2, row3], dim=0)  # (9,9)
        # Build Bd as a block column: Bd = [O3; O3; w^3] -> (9,3)
        self.Bd = torch.cat([O3, O3, w3], dim=0)
    
    def get_eta_d(self):
        """Return desired vessel pose (3,)."""
        return self.eta_d

    def get_eta_d_dot(self):
        """Return desired vessel velocity (3,)."""
        return self.eta_d_dot

    def get_eta_d_ddot(self):
        """Return desired vessel acceleration (3,)."""
        return self.eta_d_ddot

    def get_nu_d(self):
        """
        Return desired velocity in the vessel's body frame.
        This function requires a differentiable rotation; assume Rz_torch is imported.
        """
        psi = self.eta_d[-1]
        return torch.matmul(Rz_torch(psi).transpose(0,1), self.eta_d_dot)

    def set_eta_r(self, eta_r):
        """
        Set the reference pose.
        
        Parameters:
          eta_r: iterable of length 3.
        """
        self._eta_r = torch.tensor(eta_r, dtype=torch.float32, device=self._x.device)

    def update(self):
        """
        Update the reference state using:
          _x = _x + dt * (Ad @ _x + Bd @ _eta_r)
        Then update eta_d, eta_d_dot, and eta_d_ddot from _x.
        """
        x_dot = torch.matmul(self.Ad, self._x) + torch.matmul(self.Bd, self._eta_r)
        self._x = self._x + self._dt * x_dot
        self.eta_d = self._x[:3]
        self.eta_d_dot = self._x[3:6]
        self.eta_d_ddot = self._x[6:]