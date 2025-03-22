#!/usr/bin/env python3
"""
csad_torch.py

Differentiable 6 DOF DP vessel model for CSAD implemented in PyTorch.
This class replicates the functionality of the original CSAD_DP_6DOF class but
operates on torch.Tensors for use in gradient-based meta learning.

Author: [Kristian Magnus Roen/ adapted from Jan-Erik Hygen]
Date:   2025-02-17
"""

import torch
import os
import json
from  torch_core.simulator.vessels.vessel_torch import Vessel
from torch_core.utils import Rz_torch, J_torch

class CSAD_6DOF(Vessel):
    """
    Differentiable 6 DOF DP simulator model for CSAD.
    A direct PyTorch replica of the old CSAD_DP_6DOF in csad.py.
    """
    def __init__(self,
                 dt,
                 method="RK4",
                 config_file="/home/kmroen/miniconda3/envs/tensor/lib/python3.9/site-packages/mclsimpy/vessel_data/CSAD/vessel_json.json",
                 dof=6):
        """
        Parameters
        ----------
        dt : float
            Timestep for integration.
        method : str
            Integration method ("Euler" or "RK4").
        config_file : str
            JSON file with the same structure as 'vessel_json.json'.
        dof : int
            Degrees of freedom (6).
        """
        # Path to your JSON config (adjust if needed)
        DATA_DIR = "/home/kristmro/workspace/CSAD/venv/lib/python3.8/site-packages/MCSimPython/vessel_data/CSAD/"
        cfg_path = os.path.join(DATA_DIR, config_file)

        # Call the DiffVessel constructor
        super().__init__(dt=dt, method=method, config_file=cfg_path, dof=dof)

        # Load JSON
        with open(cfg_path, 'r') as f:
            data = json.load(f)

        # Rigid body and added mass
        self._Mrb = torch.tensor(data['MRB'], dtype=torch.float32)
        self._Ma  = torch.tensor(data['A'],   dtype=torch.float32)[:, :, 41]
        self._M   = self._Mrb + self._Ma
        self._Minv= torch.inverse(self._M)

        # Potential + viscous damping
        self._Dp = torch.tensor(data['B'],  dtype=torch.float32)[:, :, 41]
        self._Dv = torch.tensor(data['Bv'], dtype=torch.float32)
        self._D  = self._Dp + self._Dv
        # Increase roll damping
        self._D[3, 3] = self._D[3, 3] * 2.0

        # Restoring matrix
        self._G = torch.tensor(data['C'], dtype=torch.float32)[:, :, 0]

    def x_dot(self, x, Uc, betac, tau):
        """
        Compute time derivative of state x for 6DOF DP model.

        Parameters
        ----------
        x : torch.Tensor, shape (12,)
            Vessel state: first 6 are eta (x,y,z, roll,pitch,yaw),
                          last 6 are nu  (u,v,w, p,q,r).
        Uc : float or torch scalar
            Current speed (earth frame).
        betac : float or torch scalar
            Current direction in earth-fixed frame [rad].
        tau : torch.Tensor, shape (6,)
            External loads (thrusters, wind, etc.).

        Returns
        -------
        x_dot : torch.Tensor, shape (12,)
            Time derivative of state vector [eta_dot, nu_dot].
        """
        # Split the state
        eta = x[:6]  # (roll, pitch, yaw are indices 3,4,5)
        nu  = x[6:]

        # 1) Build earth-frame current velocity (3D)
        #    old code:  nu_cn = Uc * [cos(betac), sin(betac), 0]
        betac_t = torch.tensor(betac, dtype=torch.float32, device=x.device)
        cos_b = torch.cos(betac_t)
        sin_b = torch.sin(betac_t)
        nu_cn = Uc * torch.stack([cos_b, sin_b, torch.tensor(0.0, device=x.device, dtype=x.dtype)], dim=0)

        # 2) Rotate current into body frame via yaw: Rz(eta[5]).T @ nu_cn
        R_yaw = Rz_torch(eta[-1])       # shape (3,3)
        nu_c3 = R_yaw.transpose(0,1) @ nu_cn  # shape (3,)

        # Expand to full 6 DOF by adding zeros in the last 3 components
        nu_c = torch.cat([nu_c3, torch.zeros(3, device=x.device, dtype=x.dtype)])

        # Relative velocity
        nu_r = nu - nu_c

        # 3) Kinematics:  eta_dot = J(eta) @ nu
        #    old code calls J(eta) for a 6x6 transform
        J_ = J_torch(eta)    # shape (6,6)
        eta_dot = J_ @ nu    # shape (6,)

        # 4) Kinetics: nu_dot = M^{-1} ( tau - D@nu_r - G@eta )
        nu_dot = self._Minv @ (tau - self._D @ nu_r - self._G @ eta)

        # Return [eta_dot, nu_dot]
        return torch.cat([eta_dot, nu_dot], dim=0)