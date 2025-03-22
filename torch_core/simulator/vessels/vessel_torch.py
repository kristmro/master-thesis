#!/usr/bin/env python3
"""
vessel_torch.py

Differentiable base class for simulator vessels using PyTorch.
This class replicates the functionality of the original Vessel class,
but stores all state as torch.Tensors and uses differentiable (torch) operations.

Author: [Kristian Magnus Roen/ adapted from Jan-Erik Hygen]
Date:   2025-02-17
"""

import torch
from abc import ABC, abstractmethod
from torch_core.utils import pipi


class Vessel(ABC):
    def __init__(self, dt, method="RK4", config_file="", dof=6):
        self._config_file = config_file
        self._dof = dof
        self._dt = dt
        # Initialize state as torch tensors.
        self._M = torch.zeros((dof, dof), dtype=torch.float32)
        self._D = torch.zeros((dof, dof), dtype=torch.float32)
        self._G = torch.zeros((dof, dof), dtype=torch.float32)
        self._eta = torch.zeros(dof, dtype=torch.float32)
        self._nu = torch.zeros(dof, dtype=torch.float32)
        self._x = torch.cat([self._eta, self._nu])
        self._x_dot = torch.zeros(2*dof, dtype=torch.float32)
        if method == "Euler":
            self.int_method = self.forward_euler
        elif method == "RK4":
            self.int_method = self.RK4
        else:
            raise ValueError(f"{method} is not a valid integration method. Only 'Euler' and 'RK4' are accepted.")

    def set_eta(self, eta):
        if eta.shape != self._eta.shape:
            raise ValueError(f"eta shape {eta.shape} does not match DOF {self._dof}")
        self._eta = eta
        self._x[:self._dof] = self._eta

    def set_nu(self, nu):
        if nu.shape != self._nu.shape:
            raise ValueError(f"nu shape {nu.shape} does not match DOF {self._dof}")
        self._nu = nu
        self._x[self._dof:] = self._nu

    def get_eta(self):
        return self._eta

    def get_nu(self):
        return self._nu

    def get_x(self):
        return self._x

    def reset(self):
        self._x = torch.zeros(2*self._dof, dtype=torch.float32)
        self._x_dot = torch.zeros(2*self._dof, dtype=torch.float32)
        self._eta = torch.zeros(self._dof, dtype=torch.float32)
        self._nu = torch.zeros(self._dof, dtype=torch.float32)

    @abstractmethod
    def x_dot(self, x, Uc, beta_c, tau):
        """
        Compute the time derivative of the state vector x.

        Parameters:
            x: Tensor of shape (2*dof,)
            Uc: Scalar (current velocity)
            beta_c: Scalar (current direction in rad)
            tau: Tensor of shape (dof,) representing control forces.
        Returns:
            x_dot: Tensor of shape (2*dof,)
        """
        pass
    def integrate(self, Uc, beta_c, tau):
        x_old = self._x
        x_new = self.int_method(x_old, Uc, beta_c, tau)

        # Update internal state
        eta_new = x_new[:self._dof]
        # Wrap angles
        eta_new[3:] = pipi(eta_new[3:])

        nu_new  = x_new[self._dof:]
        self.set_eta(eta_new)
        self.set_nu(nu_new)

    def forward_euler(self, x, Uc, beta_c, tau):
        return x + self._dt * self.x_dot(x, Uc, beta_c, tau)

    def RK4(self, x, Uc, beta_c, tau):
        k1 = self.x_dot(x, Uc, beta_c, tau)
        k2 = self.x_dot(x + k1*self._dt/2, Uc, beta_c, tau)
        k3 = self.x_dot(x + k2*self._dt/2, Uc, beta_c, tau)
        k4 = self.x_dot(x + k3*self._dt, Uc, beta_c, tau)
        return x + (k1 + 2*k2 + 2*k3 + k4) * self._dt / 6