# reference_filter.py

# ----------------------------------------------------------------------------
# This code is part of the MCSimPython toolbox and repository
# Created By: Jan-Erik Hygen
# Created Date: 2023-01-30, 
# Revised: 2025-01-31 Kristian Magnus Roen   Now fitting the MC-Gym for csad.
# Tested:
# Copyright (C) 2023: NTNU, Trondheim
# Licensed under GPL-3.0-or-later
# ---------------------------------------------------------------------------

import numpy as np
from mclsimpy.utils import Rz

class ThrdOrderRefFilter():
    """Third-order reference filter for guidance. 
    
    Attributes
    ----------
    eta_d : array_like
        3D-array of desired vessel pose in NED-frame
    eta_d_dot : array_like
        3D-array of desired vessel velocity in NED-frame
    eta_d_ddot : array_like
        3D-array of desired vessel acceleration in NED-frame
    """

    def __init__(self, dt, omega=[0.2, 0.2, 0.2], initial_eta=None):
        self._dt = dt
        self.eta_d = np.zeros(3) if initial_eta is None else np.array(initial_eta)  # Start at given initial_eta
        self.eta_d_dot = np.zeros(3)
        self.eta_d_ddot = np.zeros(3)
        self._eta_r = self.eta_d.copy()
        self._x = np.concatenate([self.eta_d, self.eta_d_dot, self.eta_d_ddot], axis=None)
        self._delta = np.eye(3)
        self._w = np.diag(omega)
        O3x3 = np.zeros((3, 3))
        self.Ad = np.block([
            [O3x3, np.eye(3), O3x3],
            [O3x3, O3x3, np.eye(3)],
            [-self._w**3, -(2*self._delta + np.eye(3))@self._w**2, - (2*self._delta + np.eye(3))@self._w]
        ])
        self.Bd = np.block([
            [O3x3],
            [O3x3],
            [self._w**3]
        ])


    def get_eta_d(self):
        """Get desired pose in NED-frame."""
        return self.eta_d
    
    def get_eta_d_dot(self):
        """Get desired velocity in NED-frame."""
        return self.eta_d_dot
    
    def get_eta_d_ddot(self):
        """Get desired acceleration in NED-frame."""
        return self.eta_d_ddot

    def get_nu_d(self):
        """Get desired velocity in body-frame."""
        psi = self.eta_d[-1]
        return Rz(psi).T@self.eta_d_dot

    def set_eta_r(self, eta_r):
        """Set the reference pose.
        
        Parameters
        ----------
        eta_r : array_like
            Reference vessel pose in surge, sway and yaw.
        """
        self._eta_r = eta_r

    def update(self):
        """Update the desired position."""
        x_dot = self.Ad@self._x + self.Bd@self._eta_r
        self._x = self._x + self._dt*x_dot
        self.eta_d = self._x[:3]
        self.eta_d_dot = self._x[3:6]
        self.eta_d_ddot = self._x[6:]