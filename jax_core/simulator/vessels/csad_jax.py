"""
csad_jax.py

Implements a differentiable 6 Degrees-of-Freedom (DOF) dynamic positioning (DP) vessel model using JAX.
Loads vessel parameters and computes the state derivatives for use in simulations and gradient-based meta-learning.

Author: Kristian Magnus Roen (adapted from Jan-Erik Hygen)
Date:   2025-03-17
"""

import json
import jax
import jax.numpy as jnp
from jax_core.utils import Rz, J  # Ensure these are pure functions as well

def load_csad_parameters(config_file="/home/kmroen/miniconda3/envs/tensor/lib/python3.9/site-packages/mclsimpy/vessel_data/CSAD/vessel_json.json"):
    """
    Load the vessel parameters from a JSON file and construct the system matrices.
    
    Returns a dictionary with keys:
      - Mrb: Rigid-body mass matrix.
      - Ma: Added mass matrix (initially selected at a specific index).
      - M: Total mass matrix (Mrb + Ma).
      - Minv: Inverse of total mass matrix.
      - Dp: Hydrodynamic damping matrix part from potential flow.
      - Dv: Viscous damping matrix.
      - D: Total damping matrix (Dp + Dv), with an adjustment at index (3,3).
      - G: Restoring matrix.
    """
    with open(config_file, "r") as f:
        data = json.load(f)

    Mrb = jnp.asarray(data["MRB"])
    Ma = jnp.asarray(data["A"])[:, :, 41]
    M = Mrb + Ma
    Minv = jnp.linalg.inv(M)

    Dp = jnp.asarray(data["B"])[:, :, 41]
    Dv = jnp.asarray(data["Bv"])
    D = Dp + Dv
    D = D.at[3, 3].set(D[3, 3] * 2)

    G = jnp.asarray(data["C"])[:, :, 0]

    params = {
        "Mrb": Mrb,
        "Ma": Ma,
        "M": M,
        "Minv": Minv,
        "Dp": Dp,
        "Dv": Dv,
        "D": D,
        "G": G,
    }
    return params

def csad_x_dot(x, Uc, betac, tau, params):
    """
    Compute the time derivative of the state for the 6 DOF vessel model.
    
    Parameters:
      x: A 12-element state vector [eta, nu], where eta (positions/orientations) 
         and nu (velocities) are 6-element vectors.
      Uc: Current speed.
      betac: Current direction (in radians).
      tau: External forces/torques (6-element vector).
      params: Dictionary containing system matrices (from load_csad_parameters or set_hydrod_parameters).
    
    Returns:
      dx/dt: A 12-element vector combining eta_dot and nu_dot.
    """
    eta = x[:6]
    nu  = x[6:]
    
    # Compute the current component in the inertial frame
    nu_cn = Uc * jnp.array([jnp.cos(betac), jnp.sin(betac), 0.0])
    
    # Rotate current into the body-fixed frame using the yaw angle (eta[-1])
    nu_c = jnp.transpose(Rz(eta[-1])) @ nu_cn
    # Insert zeros for the rotational DOFs (assuming indices 3,4,5 correspond to rotations)
    nu_c = jnp.insert(nu_c, jnp.array([3, 3, 3]), 0)
    
    # Relative velocity (subtracting current effects)
    nu_r = nu - nu_c
    
    # Kinematics: transform body velocities to inertial rates (using a transformation matrix J)
    eta_dot = J(eta) @ nu
    
    # Kinetics: acceleration computed from external forces, damping, and restoring forces
    nu_dot = params["Minv"] @ (tau - params["D"] @ nu_r - params["G"] @ eta)
    
    return jnp.concatenate([eta_dot, nu_dot])

def set_hydrod_parameters(freq, params, config_file="/home/kmroen/miniconda3/envs/tensor/lib/python3.9/site-packages/mclsimpy/vessel_data/CSAD/vessel_json.json"):
    """
    Update hydrodynamic parameters for a given frequency (or per-DOF frequencies).
    
    Parameters:
      freq: A scalar frequency or a 1D array of length 6 (one per DOF).
      params: Dictionary containing initial parameters (including config_file).
    
    Returns:
      new_params: A new parameters dictionary with updated hydrodynamic matrices:
                  - Ma, Dp, M, Minv, and D.
    """
    # Ensure freq is a JAX array
    if not isinstance(freq, (list, tuple, jnp.ndarray)):
        freq = jnp.array([freq])
    else:
        freq = jnp.array(freq)

    # Check dimensions: if multiple frequencies, expect one per DOF (6)
    if freq.ndim == 1 and (freq.shape[0] > 1 and freq.shape[0] != 6):
        raise ValueError(f"freq must be a scalar or have shape (6,), got shape {freq.shape}.")

    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    freqs = jnp.asarray(config_data['freqs'])
    
    if freq.ndim == 1 and freq.shape[0] == 1:
        # Single frequency: choose index minimizing absolute difference
        freq_indx = jnp.argmin(jnp.abs(freqs - freq))
    else:
        # Multiple frequencies: per DOF index (assumes freq is (6,))
        freq_indx = jnp.argmin(jnp.abs(freqs - freq[:, None]), axis=1)
    
    all_dof = jnp.arange(6)
    # Gather new added mass and damping matrices using the computed indices
    Ma = jnp.asarray(config_data['A'])[:, all_dof, freq_indx]
    Dp = jnp.asarray(config_data['B'])[:, all_dof, freq_indx]
    
    M = params["Mrb"] + Ma
    Minv = jnp.linalg.inv(M)
    D = params["Dv"] + Dp

    new_params = dict(params)
    new_params.update({
        "Ma": Ma,
        "Dp": Dp,
        "M": M,
        "Minv": Minv,
        "D": D,
    })
    return new_params
