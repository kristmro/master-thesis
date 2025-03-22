#!/usr/bin/env python3
"""
utils.py
TODO: remove all the numpy dependencies and use torch instead

Differentiable utility functions for marine simulations using PyTorch.
Contains functions like Rz_torch, diff_J, pipi, to_positive_angle, and
differentiable_interp1d with an extra "dim" parameter, as well as three2sixDOF
and six2threeDOF conversions.
"""

import torch
import math
import numpy as np

def Rx_torch(phi):
    """3x3 rotation matrix about x-axis."""
    c = torch.cos(phi)
    s = torch.sin(phi)
    return torch.stack([
        torch.stack([torch.tensor(1.0, dtype=phi.dtype, device=phi.device), torch.tensor(0.0, dtype=phi.dtype, device=phi.device), torch.tensor(0.0, dtype=phi.dtype, device=phi.device)]),
        torch.stack([torch.tensor(0.0, dtype=phi.dtype, device=phi.device),              c,                          -s]),
        torch.stack([torch.tensor(0.0, dtype=phi.dtype, device=phi.device),              s,                           c])
    ])

def Ry_torch(theta):
    """3x3 rotation matrix about y-axis."""
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.stack([
        torch.stack([     c, torch.tensor(0.0, dtype=theta.dtype, device=theta.device),     s]),
        torch.stack([torch.tensor(0.0, dtype=theta.dtype, device=theta.device), torch.tensor(1.0, dtype=theta.dtype, device=theta.device), torch.tensor(0.0, dtype=theta.dtype, device=theta.device)]),
        torch.stack([    -s, torch.tensor(0.0, dtype=theta.dtype, device=theta.device),     c])
    ])

def Rz_torch(psi):
    """3x3 rotation matrix about z-axis."""
    c = torch.cos(psi)
    s = torch.sin(psi)
    return torch.stack([
        torch.stack([     c, -s, torch.tensor(0.0, dtype=psi.dtype, device=psi.device)]),
        torch.stack([     s,  c, torch.tensor(0.0, dtype=psi.dtype, device=psi.device)]),
        torch.stack([torch.tensor(0.0, dtype=psi.dtype, device=psi.device), torch.tensor(0.0, dtype=psi.dtype, device=psi.device), torch.tensor(1.0, dtype=psi.dtype, device=psi.device)])
    ])
def Rz_torch_2(psi):
    """
    Compute a 3x3 rotation matrix about the z-axis in batched fashion.
    psi can be of shape (...). The output will have shape (..., 3, 3).
    """
    c = torch.cos(psi)
    s = torch.sin(psi)
    zero = torch.zeros_like(c)
    one  = torch.ones_like(c)
    # Stack each row along the last dimension.
    row1 = torch.stack([c, -s, zero], dim=-1)  # (..., 3)
    row2 = torch.stack([s,  c, zero], dim=-1)   # (..., 3)
    row3 = torch.stack([zero, zero, one], dim=-1) # (..., 3)
    # Stack rows along a new dimension (penultimate) to form matrices.
    return torch.stack([row1, row2, row3], dim=-2)  # (..., 3, 3)


def Rzyx_torch(phi, theta, psi):
    """
    Composite rotation matrix: Rz(psi)*Ry(theta)*Rx(phi), matching 'Rzyx' in old code.
    We assume order phi->theta->psi means Rz(psi) * Ry(theta) * Rx(phi).
    """
    return Rz_torch(psi) @ Ry_torch(theta) @ Rx_torch(phi)

def Tzyx_torch(phi, theta, psi):
    """
    3x3 Euler angle rate transform matrix matching 'Tzyx(eta)' in old code:
       T( phi, theta ) = 
         [[1,  sin(phi)*tan(theta),  cos(phi)*tan(theta)],
          [0,  cos(phi),            -sin(phi)],
          [0,  sin(phi)/cos(theta),  cos(phi)/cos(theta)]]
    The 'psi' does not appear in the expressions, 
    but we keep the same signature for consistency.
    """
    sinp = torch.sin(phi)
    cosp = torch.cos(phi)
    sint = torch.sin(theta)
    cost = torch.cos(theta)
    tant = torch.tan(theta)

    return torch.stack([
        torch.stack([torch.tensor(1.0, dtype=phi.dtype, device=phi.device),  sinp*tant,   cosp*tant]),
        torch.stack([torch.tensor(0.0, dtype=phi.dtype, device=phi.device),      cosp,         -sinp]),
        torch.stack([torch.tensor(0.0, dtype=phi.dtype, device=phi.device),  sinp/cost,    cosp/cost])
    ])

def J_torch(eta):
    """
    6x6 transform matrix matching the old 'J(eta)' function:
      J(eta) = [[Rzyx(eta[3],eta[4],eta[5]),  0],
                [0,                           Tzyx(eta[3],eta[4],eta[5])]]
    where eta is [x, y, z, roll, pitch, yaw].
    """
    phi   = eta[3]
    theta = eta[4]
    psi   = eta[5]

    R_part = Rzyx_torch(phi, theta, psi)          # shape (3,3)
    T_part = Tzyx_torch(phi, theta, psi)          # shape (3,3)

    # Build top row block [ R , 0 ]
    top = torch.cat([
        R_part, 
        torch.zeros((3,3), dtype=eta.dtype, device=eta.device)
    ], dim=1)  # shape (3,6)

    # Build bottom row block [ 0 , T ]
    bottom = torch.cat([
        torch.zeros((3,3), dtype=eta.dtype, device=eta.device),
        T_part
    ], dim=1)  # shape (3,6)

    # Full 6x6
    return torch.cat([top, bottom], dim=0)


def torch_lininterp_1d(x, y, x_new, axis=0, left_fill=None, right_fill=None):
    """
    Minimal 1D linear interpolation in PyTorch, replicating bounds_error=False, fill_value=(..., ...).

    - x: shape(N,)  (Must be sorted in ascending order.)
    - y: shape(N,...) if axis=0, or (...,N) if axis=-1
    - x_new: shape(*)
    Returns shape of x_new plus shape of y minus axis dimension.
    """
    if x.dim() != 1:
        raise ValueError("Input x must be a 1D Tensor.")

    # If we need to permute the axis=0 to front, handle that:
    if axis != 0:
        dims = list(range(y.dim()))
        if axis < 0:
            axis = y.dim() + axis
        perm = [axis] + dims[:axis] + dims[axis+1:]
        y_perm = y.permute(perm)
    else:
        y_perm = y

    N = x.shape[0]
    shape_rest = y_perm.shape[1:]  # everything but the first dimension
    M = 1
    for s in shape_rest:
        M *= s
    y_flat = y_perm.reshape(N, M)

    x_new_flat = x_new.flatten()
    K = x_new_flat.shape[0]

    # bucketize => index in [1..N-1]
    idx = torch.bucketize(x_new_flat, x)
    idx = torch.clamp(idx, 1, N-1)

    x0 = x[idx-1]
    x1 = x[idx]
    denom = (x1 - x0).clone()
    denom[denom==0] = 1e-9
    w = (x_new_flat - x0)/denom

    out_flat = torch.empty((K, M), dtype=y.dtype, device=y.device)
    for col in range(M):
        col_y = y_flat[:, col]
        y0 = col_y[idx-1]
        y1 = col_y[idx]
        interped = y0 + (y1-y0)*w
        out_flat[:, col] = interped

    # OOB fill
    left_mask  = x_new_flat < x[0]
    right_mask = x_new_flat > x[-1]
    if left_fill is not None:
        out_flat[left_mask] = left_fill
    else:
        # If not specified, clamp to first value
        for col in range(M):
            out_flat[left_mask, col] = y_flat[0, col]
    if right_fill is not None:
        out_flat[right_mask] = right_fill
    else:
        # If not specified, clamp to last value
        for col in range(M):
            out_flat[right_mask, col] = y_flat[-1, col]

    out_reshaped = out_flat.view(*x_new.shape, *shape_rest)
    if axis != 0:
        inv_perm = [0]*len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out_final = out_reshaped.permute(inv_perm)
    else:
        out_final = out_reshaped
    return out_final
# --------------------------------------------------------------------------
# Utility angle functions
# --------------------------------------------------------------------------
def pipi(theta):
    """Map angle to [-pi, pi)."""
    return torch.remainder(theta + math.pi, 2*math.pi) - math.pi

def to_positive_angle(theta):
    """Map angle from [-pi, pi) to [0, 2*pi)."""
    return torch.where(theta < 0, theta + 2*math.pi, theta)

def three2sixDOF(v):
    """
    Convert a 3DOF vector or matrix to 6DOF.
    If v is a vector of shape (3,), returns tensor of shape (6,) with mapping:
      [v0, v1, 0, 0, 0, v2].
    If v is a matrix of shape (3,3), flatten and embed into a 6x6 matrix.
    """
    if v.dim() == 1:
        return torch.tensor([v[0], v[1], 0.0, 0.0, 0.0, v[2]], dtype=v.dtype, device=v.device)
    elif v.dim() == 2:
        flat = v.flatten()
        part1 = flat[0:2]
        part2 = torch.zeros(3, dtype=v.dtype, device=v.device)
        part3 = flat[2:5]
        part4 = torch.zeros(3, dtype=v.dtype, device=v.device)
        part5 = flat[5:6]
        part6 = torch.zeros(18, dtype=v.dtype, device=v.device)
        part7 = flat[6:8]
        part8 = torch.zeros(3, dtype=v.dtype, device=v.device)
        part9 = flat[8:9]
        out_flat = torch.cat([part1, part2, part3, part4, part5, part6, part7, part8, part9])
        return out_flat.view(6,6)
    else:
        raise ValueError("Input v must be 1D or 2D tensor.")


def six2threeDOF(v):
    """
    Convert a 6DOF vector or matrix to 3DOF.
    - If v is a vector of shape (6,), returns the elements at indices [0, 1, 5] → shape (3,).
    - If v is a matrix of shape (T,6) with T != 6, returns the columns [0, 1, 5] → shape (T,3).
    - If v is a 6x6 matrix, returns the submatrix rows/cols [0, 1, 5] → shape (3,3).
    """
    # If v is a NumPy array, convert to torch tensor
    if isinstance(v, np.ndarray):
        v = torch.from_numpy(v)

    if v.dim() == 1:
        # Expect v.shape == (6,)
        if v.size(0) != 6:
            raise ValueError(f"Expected 6 elements in 1D, got shape {tuple(v.shape)}")
        return v[[0, 1, 5]]

    elif v.dim() == 2:
        rows, cols = v.shape
        if rows == 6 and cols == 6:
            # 6x6 → return submatrix
            return v[[0, 1, 5]][:, [0, 1, 5]]
        elif cols == 6:
            # (T,6) → return columns [0,1,5]
            return v[:, [0, 1, 5]]
        else:
            raise ValueError(f"Expected shape (T,6) or (6,6), got {v.shape} instead.")

    else:
        raise ValueError(f"Input must be 1D or 2D, but got dim={v.dim()}.")