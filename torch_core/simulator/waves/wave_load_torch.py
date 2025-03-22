#!/usr/bin/env python3
"""
wave_load_torch.py

Fully differentiable PyTorch version of your original WaveLoad class.
It replicates the same logic, reading the same config_file, and uses
PyTorch interpolation methods (e.g. torch_lininterp_1d) to replace the old scipy.interp1d.

Author: [Kristian Magnus Roen/ adapted from Jan-Erik Hygen]
Date:   2025-02-17
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import json
import os
from torch_core.utils import pipi, to_positive_angle, torch_lininterp_1d

################################################################
# The main differentiable WaveLoad class
################################################################

class WaveLoad(torch.nn.Module):
    """
    Differentiable wave load module in PyTorch, replicating the old wave_loads.py logic.

    This class calculates first- and second-order wave loads:
      - 1st order from vessel force RAOs
      - 2nd order from drift QTF matrices
    using data from a config file that includes forceRAO, driftfrc, freq, headings, etc.
    """
    QTF_METHODS = ["Newman", "geo-mean"]

    def __init__(self,
                 wave_amps,           # array-like (N,) wave amplitude
                 freqs,               # array-like (N,) wave frequencies in rad/s
                 eps,                 # array-like (N,) random phases
                 angles,              # array-like (N,) wave directions
                 config_file,         # vessel JSON path
                 rho=1025,
                 g=9.81,
                 dof=6,
                 depth=100.0,
                 deep_water=True,
                 qtf_method="Newman",
                 qtf_interp_angles=True,
                 interpolate=True):
        super().__init__()

        # 1) Store wave data as Buffers => not trainable, but moved with .to(device).
        self.register_buffer("_amp",    torch.tensor(wave_amps, dtype=torch.float32))
        self.register_buffer("_freqs",  torch.tensor(freqs,     dtype=torch.float32))
        self.register_buffer("_eps",    torch.tensor(eps,       dtype=torch.float32))
        self.register_buffer("_angles", torch.tensor(angles,    dtype=torch.float32))

        self._N = self._amp.shape[0]
        self._rho = rho
        self._g = g
        self._dof = dof
        self._depth = depth
        self._deep_water = deep_water
        self._qtf_method = qtf_method
        self._qtf_interp_angles = qtf_interp_angles
        self._interpolate = interpolate

        # 2) Load vessel config for drift & RAO data
        with open(config_file, 'r') as f:
            vessel_params = json.load(f)
        self._params = vessel_params

        # headings/freqs from config => torch
        qtf_angles = torch.tensor(vessel_params['headings'], dtype=torch.float32)
        qtf_freqs  = torch.tensor(vessel_params['freqs'],    dtype=torch.float32)
        self.register_buffer("_qtf_angles", qtf_angles)
        self.register_buffer("_qtf_freqs",  qtf_freqs)

        # 3) Wave numbers
        if self._deep_water:
            # k = w^2 / g
            self.register_buffer("_k", (self._freqs**2)/self._g)
        else:
            # iterative approach
            k_list = []
            for wval in self._freqs:
                w_ = wval.item()
                k_old = w_**2 / self._g
                k_new = w_**2/(self._g*math.tanh(k_old*self._depth))
                itcount=0
                while abs(k_new - k_old)>1e-5 and itcount<50:
                    k_old=k_new
                    k_new=w_**2/(self._g*math.tanh(k_old*self._depth))
                    itcount+=1
                k_list.append(k_new)
            self.register_buffer("_k", torch.tensor(k_list, dtype=torch.float32))

        # difference freq => shape(N,N)
        W = self._freqs.view(-1,1) - self._freqs.view(1,-1)
        # difference phase => shape(N,N)
        P = self._eps.view(-1,1) - self._eps.view(1,-1)
        self.register_buffer("_W", W)
        self.register_buffer("_P", P)

        # 4) Build Q from driftfrc => shape(6, freq_count, heading_count, possibly velocity index)
        drift_amp_np = np.array(vessel_params['driftfrc']['amp'], dtype=np.float32)
        drift_amp_t  = torch.tensor(drift_amp_np, dtype=torch.float32)
        # e.g., (6, freq_count, heading_count, nVel). Pick index=0 if needed
        if drift_amp_t.dim() == 4:
            drift_amp_t = drift_amp_t[:, :, :, 0]
        Q = self._build_qtf_6dof(self._freqs, qtf_angles, qtf_freqs, drift_amp_t,
                                 method=self._qtf_method,
                                 interpolate=self._interpolate,
                                 qtf_interp_angles=self._qtf_interp_angles)
        self.register_buffer("_Q", Q)

        # 5) Force RAOs from 'forceRAO'
        force_amp_full   = torch.tensor(vessel_params['forceRAO']['amp'],   dtype=torch.float32)
        force_phase_full = torch.tensor(vessel_params['forceRAO']['phase'], dtype=torch.float32)
        # shape => (6, freq_count, heading_count, something). Slice => 0
        force_amp   = force_amp_full[:, :, :, 0]
        # convert from degrees => radians
        force_phase = force_phase_full[:, :, :, 0] * math.pi/180.0

        fAmp, fPhase = self._build_force_raos(force_amp, force_phase, qtf_freqs)
        self.register_buffer("_forceRAOamp",   fAmp)
        self.register_buffer("_forceRAOphase", fPhase)

    ########################################################################
    # The "forward" and the load computations
    ########################################################################
    def forward(self, time, eta):
        """
        Equivalent to old __call__:
          returns total wave load = first_order + second_order
        """
        tau_wf = self.first_order_loads(time, eta)
        tau_sv = self.second_order_loads(time, eta[-1])
        return tau_wf + tau_sv

    def first_order_loads(self, t, eta):
        """
        Compute 1st-order wave loads by summing wave components:
          tau_wf = sum_j [ rao_amp[:,j] * cos(... - rao_phase[:,j] ) * wave_amp[j] ]
        """
        # 1) relative angles
        rel_angle = self._relative_incident_angle(eta[-1])  # shape(N,)
        # 2) rao amplitude & phase => shape(6,N)
        rao_amp, rao_phase = self._rao_interp(rel_angle)

        # 3) wave phase => shape(N,)
        x, y = eta[0].item(), eta[1].item()
        wave_phase = (self._freqs * t
                      - self._k * x * torch.cos(self._angles)
                      - self._k * y * torch.sin(self._angles)
                      - self._eps)
        # shape => (6,N)
        arg_matrix = wave_phase.unsqueeze(0) - rao_phase
        cos_val = torch.cos(arg_matrix)
        contrib = rao_amp * cos_val
        tau_wf = torch.matmul(contrib, self._amp.unsqueeze(-1)).squeeze(-1) # shape(6,)
        return tau_wf

    def second_order_loads(self, t, heading):
        """
        Compute 2nd-order slow-drift from QTF => real( amp * Q e^{i(Wt+P)} * amp ).
        We pick the Q row for the nearest heading (0..359 deg).
        """
        rel_angle = self._relative_incident_angle(heading)
        mean_angle = torch.mean(rel_angle)
        angles_1deg = torch.linspace(0, 2*math.pi, 360, device=rel_angle.device)
        diffs = torch.abs(angles_1deg - mean_angle)
        heading_index = torch.argmin(diffs)
        # pick Q => shape(6,N,N)
        Q_sel = self._Q[:, heading_index, :, :]

        # e^{i(W t + P)} => shape(N,N)
        amp_cplx = self._amp.to(torch.complex64)
        exp_term = torch.exp(1j*(self._W*t + self._P))  # shape(N,N)

        out_vals=[]
        for d in range(self._dof):
            Qd = Q_sel[d] # shape(N,N)
            mat = Qd * exp_term
            val = amp_cplx.unsqueeze(0) @ mat @ amp_cplx.unsqueeze(1)
            out_vals.append(torch.real(val[0,0]))
        tau_sv = torch.stack(out_vals, dim=0)
        return tau_sv

    ########################################################################
    # The angle-based interpolation for first-order RAOs
    ########################################################################
    def _rao_interp(self, rel_angle):
        """
        Old code uses a discrete bin approach:
          index_lb = argmin( difference vs floor(...) )
          index_ub = ...
          scale = ...
        to find RAO amplitude & phase for each wave component (N).
        We produce arrays shape(6,N) => rao_amp, rao_phase.

        This matches your old code's approach, but it's not fully smooth. 
        """
        # Convert self._qtf_angles to deg
        qtf_deg = self._qtf_angles * 180./math.pi  # shape(H,)
        # rel_angle => shape(N,)
        rel_deg  = rel_angle * 180./math.pi        # shape(N,)

        # We do floor( rel_deg/10 ) * 10 => LB index => UB= +1
        # In the old code, you do:
        # index_lb= argmin(| deg(_qtf_angles) - floor(rel_deg/10)*10 |, axis=1)
        # We replicate that logic in PyTorch if possible. 
        # But to keep it simple, let's do it in NumPy or a half-Torch approach.
        # For a purely direct replica, we keep it piecewise. 
        # We'll do the same approach as your old code. For each wave i:
        device = rel_angle.device
        N = rel_angle.shape[0]

        # We expand each wave's angle => pick bin center => find nearest in qtf_deg
        # Let's do it the old way with a short PyTorch loop.
        # We'll store => index_lb, index_ub for each wave i

        # Convert to degrees, floor to nearest 10, etc.
        # E.g. old code does: 
        #   index_lb = np.argmin( np.abs( deg(_qtf_angles) - floor(deg(rel_angle)/10)*10 ), axis=1)
        # We'll do a direct gather approach in Torch.

        # We'll need a 1D array for qtf_deg. Then we compare each wave angle to each qtf_deg entry => find min
        # Then we do +1 for UB. 
        qtf_deg_np = qtf_deg.detach().cpu().numpy()
        rel_deg_np = rel_deg.detach().cpu().numpy()
        index_lb = []
        index_ub = []
        for i in range(N):
            bin_center = math.floor(rel_deg_np[i]/10.)*10.
            # find closest in qtf_deg => argmin(|qtf_deg - bin_center|)
            diffs = np.abs(qtf_deg_np - bin_center)
            lb_i  = int(np.argmin(diffs))
            # ub => +1 or 0 if lb_i==len(qtf_deg)-1
            if lb_i< (len(qtf_deg_np)-1):
                ub_i= lb_i+1
            else:
                ub_i= 0
            index_lb.append(lb_i)
            index_ub.append(ub_i)
        index_lb = torch.tensor(index_lb, dtype=torch.long, device=device)
        index_ub = torch.tensor(index_ub, dtype=torch.long, device=device)

        # gather => shape(6,N,1)
        # self._forceRAOamp => shape(6, Nfreq, Nheading)
        # but we want => shape(6,Nwave,Nheading?). 
        # Actually, we want => shape(6,N)
        # We'll pick freq index = wave i => that is range(N).
        freq_ind = torch.arange(N, device=device)  # 0..N-1

        # gather op in Torch => we can do advanced indexing
        # out_amp_lb = self._forceRAOamp[:, freq_ind, index_lb]
        # but freq_ind, index_lb both shape(N,). 
        # We do a loop or we replicate the old approach in a vector way.

        # We'll do loops for clarity:
        rao_amp_lb   = torch.zeros((self._dof, N), dtype=torch.float32, device=device)
        rao_phase_lb = torch.zeros_like(rao_amp_lb)
        rao_amp_ub   = torch.zeros_like(rao_amp_lb)
        rao_phase_ub = torch.zeros_like(rao_amp_lb)

        for i in range(N):
            f_i  = freq_ind[i].item()
            lb_i = index_lb[i].item()
            ub_i = index_ub[i].item()
            rao_amp_lb[:, i]   = self._forceRAOamp[:, f_i, lb_i]
            rao_phase_lb[:, i] = self._forceRAOphase[:, f_i, lb_i]
            rao_amp_ub[:, i]   = self._forceRAOamp[:, f_i, ub_i]
            rao_phase_ub[:, i] = self._forceRAOphase[:, f_i, ub_i]

        # scale => shape(N,)
        # old => scale= (pipi(rel_angle - theta1)/pipi(theta2 - theta1)) 
        # We'll replicate directly. 
        theta1 = self._qtf_angles[index_lb] # shape(N,)
        theta2 = self._qtf_angles[index_ub] # shape(N,)
        diff_t = pipi(theta2 - theta1)
        diff_t[diff_t==0] = 1e-9
        numerator = pipi(rel_angle - theta1)
        scale = numerator/diff_t  # shape(N,)

        # shape => (dof,N)
        scale_2d = scale.unsqueeze(0).expand(self._dof, -1)
        rao_amp   = rao_amp_lb + (rao_amp_ub - rao_amp_lb)*scale_2d
        rao_phase = rao_phase_lb + (rao_phase_ub - rao_phase_lb)*scale_2d

        return rao_amp, rao_phase

    @staticmethod
    def _build_qtf_6dof(wave_freqs, qtf_headings, qtf_freqs, drift_amp,
                        method="Newman", interpolate=True, qtf_interp_angles=True):
        """
        Build the full 2nd-order QTF => shape(6, 360, N, N).
        This matches the old _full_qtf_6dof logic with freq & angle interpolation,
        then Newman or geo-mean approximation.
        """
        wave_freqs = wave_freqs.to(torch.float32)
        qtf_freqs  = qtf_freqs.to(torch.float32)
        drift_amp  = drift_amp.to(torch.float32)

        dof=6
        N = wave_freqs.shape[0]

        # 1) freq interpolation => produce Qdiag(6,N,H)
        if interpolate:
            # insert freq=0 if wave_freqs[0]<qtf_freqs[0]
            if wave_freqs[0].item()<qtf_freqs[0].item():
                zero_freq= torch.tensor([0.0], dtype=torch.float32, device=qtf_freqs.device)
                qtf_freqs_mod= torch.cat([zero_freq, qtf_freqs], dim=0)
                zero_drift   = torch.zeros((dof,1, drift_amp.shape[2]),
                                           dtype=drift_amp.dtype, device=drift_amp.device)
                drift_amp_mod= torch.cat([zero_drift, drift_amp], dim=1)
            else:
                qtf_freqs_mod= qtf_freqs
                drift_amp_mod= drift_amp
            Qdiag= torch.zeros((dof,N, drift_amp_mod.shape[2]),
                               dtype=drift_amp.dtype, device=drift_amp.device)
            for d in range(dof):
                for h in range(drift_amp_mod.shape[2]):
                    Qdiag[d,:,h] = torch_lininterp_1d(
                        qtf_freqs_mod, drift_amp_mod[d,:,h], wave_freqs,
                        left_fill=drift_amp_mod[d,0,h], right_fill=drift_amp_mod[d,-1,h]
                    )
        else:
            freq_idx=[]
            qtf_freqs_np= qtf_freqs.cpu().numpy()
            wave_freqs_np= wave_freqs.cpu().numpy()
            for w_ in wave_freqs_np:
                idx= (np.abs(qtf_freqs_np-w_)).argmin()
                freq_idx.append(idx)
            freq_idx= torch.tensor(freq_idx, dtype=torch.long, device=drift_amp.device)
            Qdiag= torch.zeros((dof,N, drift_amp.shape[2]), dtype=drift_amp.dtype, device=drift_amp.device)
            for d in range(dof):
                for iN in range(N):
                    Qdiag[d,iN] = drift_amp[d, freq_idx[iN], :]

        # 2) angle interpolation => produce shape(6,360,N)
        angles_1deg= torch.linspace(0, 2*math.pi, 360, device=Qdiag.device)
        if qtf_interp_angles:
            Qdiag2= torch.zeros((dof,N,360), dtype=Qdiag.dtype, device=Qdiag.device)
            for d in range(dof):
                for iN in range(N):
                    Qdiag2[d,iN] = torch_lininterp_1d(
                        qtf_headings, Qdiag[d,iN,:], angles_1deg,
                        left_fill=Qdiag[d,iN,0], right_fill=Qdiag[d,iN,-1]
                    )
            Qdiag_final= Qdiag2.permute(0,2,1) # shape(6,360,N)
            M= 360
        else:
            Qdiag_final= Qdiag.permute(0,2,1) # shape(6,H,N)
            M= qtf_headings.shape[0]

        Q_4d= torch.zeros((dof,M,N,N), dtype=Qdiag_final.dtype, device=Qdiag_final.device)
        for d in range(dof):
            for iHead in range(M):
                qvals= Qdiag_final[d,iHead,:]
                a= qvals.unsqueeze(0).expand(N,N)
                b= qvals.unsqueeze(1).expand(N,N)
                if method.lower()=="newman":
                    Q_4d[d,iHead]= 0.5*(a+b)
                elif method.lower()=="geo-mean":
                    sign_a= torch.sign(a)
                    sign_b= torch.sign(b)
                    same_sign= (sign_a==sign_b)
                    ab= a*b
                    val= sign_a*torch.sqrt(torch.abs(ab))
                    val= torch.where(same_sign, val, torch.zeros_like(val))
                    Q_4d[d,iHead]= val
                else:
                    raise ValueError(f"Unknown QTF method: {method}")

        # old fix => Q[5]= Q[2], Q[2]=0
        Q_4d[5] = Q_4d[2].clone()
        Q_4d[2] = 0.0
        return Q_4d

    def _build_force_raos(self, force_amp, force_phase, freq_cfg):
        """
        Builds the 1st-order RAO amplitude & phase w.r.t. self._freqs.
        This matches the old _set_force_raos() approach with interpolation or nearest freq index.
        """
        dof, freq_count, heading_count = force_amp.shape
        N = self._N
        device= force_amp.device

        out_amp   = torch.zeros((dof,N, heading_count), dtype=torch.float32, device=device)
        out_phase = torch.zeros_like(out_amp)

        if self._interpolate:
            for d in range(dof):
                for h in range(heading_count):
                    yA= force_amp[d,:,h]
                    yP= force_phase[d,:,h]
                    interpA= torch_lininterp_1d(freq_cfg, yA, self._freqs,
                                                left_fill=yA[0], right_fill=yA[-1])
                    interpP= torch_lininterp_1d(freq_cfg, yP, self._freqs,
                                                left_fill=yP[0], right_fill=yP[-1])
                    out_amp[d,:,h]   = interpA
                    out_phase[d,:,h] = interpP
        else:
            freq_cfg_np= freq_cfg.cpu().numpy()
            freq_ind=[]
            for w_ in self._freqs.cpu().numpy():
                idx= np.argmin(np.abs(freq_cfg_np- w_))
                freq_ind.append(idx)
            freq_ind= np.array(freq_ind, dtype=int)
            for d in range(dof):
                for iN in range(N):
                    out_amp[d,iN]= force_amp[d,freq_ind[iN], :]
                    out_phase[d,iN]= force_phase[d,freq_ind[iN], :]

        return out_amp, out_phase

    def _relative_incident_angle(self, heading):
        """
        The relative wave incident angle for each wave component.
        Gamma = 0.0 means following sea, gamma=180 => head sea, etc.
        """
        # angles is self._angles
        raw = pipi(self._angles - heading)   # shape (N,) if heading is scalar
        return to_positive_angle(raw)

    ###############################################
    # Class method to fetch list of QTF methods
    ###############################################
    @classmethod
    def get_methods(cls):
        return cls.QTF_METHODS