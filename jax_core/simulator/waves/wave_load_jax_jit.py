"""
wave_load_jax_jit.py

Implements differentiable wave load calculations for vessel simulations in JAX.
This version is refactored so that all runtime data (used in JIT/vmap) is stored
in a PyTree-friendly dataclass. The mathematical meaning is preserved.

Author: Kristian Magnus Roen (modified)
Date:   2025-03-20
"""

import json
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import lax, vmap
from jax_core.utils import to_positive_angle, pipi

# -------------------------------------------------------------------
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class WaveLoad:
    N: int
    amp: jnp.ndarray
    freqs: jnp.ndarray
    eps: jnp.ndarray
    angles: jnp.ndarray
    depth: float
    qtf_angles: jnp.ndarray
    qtf_freqs: jnp.ndarray
    g: float
    k: jnp.ndarray
    rho: float
    W: jnp.ndarray
    P: jnp.ndarray
    Q: jnp.ndarray
    forceRAOamp: jnp.ndarray
    forceRAOphase: jnp.ndarray

    # Define flattening (only numerical arrays/scalars are children)
    def tree_flatten(self):
        children = (
            self.amp, self.freqs, self.eps, self.angles, self.qtf_angles,
            self.qtf_freqs, self.k, self.W, self.P, self.Q,
            self.forceRAOamp, self.forceRAOphase
        )
        aux_data = (self.N, self.depth, self.g, self.rho)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            N=aux_data[0],
            amp=children[0],
            freqs=children[1],
            eps=children[2],
            angles=children[3],
            depth=aux_data[1],
            qtf_angles=children[4],
            qtf_freqs=children[5],
            g=aux_data[2],
            k=children[6],
            rho=aux_data[3],
            W=children[7],
            P=children[8],
            Q=children[9],
            forceRAOamp=children[10],
            forceRAOphase=children[11]
        )

# -------------------------------------------------------------------
# Initialization function: read configuration and pack into WaveLoad.
def init_wave_load(wave_amps, freqs, eps, angles, config_file,
                   rho=1025, g=9.81, dof=6, depth=100, deep_water=True,
                   qtf_method="Newman", qtf_interp_angles=True, interpolate=True):
    """
    Initialize all wave-load parameters and precompute arrays.
    (Same math as before; note that any string arguments are only used here.)
    """
    with open(config_file, "r") as f:
        vessel_params = json.load(f)
    
    N = wave_amps.shape[0]
    amp    = wave_amps
    freqs  = freqs
    eps    = eps
    angles = angles
    
    # Precomputed values from vessel config (convert to JAX arrays)
    qtf_angles   = jnp.asarray(vessel_params["headings"])
    qtf_freqs    = jnp.asarray(vessel_params["freqs"])
    # (Other config info is used below but not stored in the final structure)
    
    # Wave number calculation
    k = freqs**2 / g
    if not deep_water:
        k = wave_number(freqs, depth, g)
    
    # Difference-frequency and phase matrices
    W = freqs[:, None] - freqs
    P = eps[:, None]   - eps
    
    # Compute full QTF matrices
    Q, qtf_angles_out = full_qtf_6dof(qtf_angles,
                                      qtf_freqs,
                                      jnp.asarray(vessel_params["driftfrc"]["amp"])[:, :, :, 0],
                                      freqs,
                                      method=qtf_method,
                                      interpolate=interpolate,
                                      qtf_interp_angles=qtf_interp_angles)
    
    # Force RAO data (first-order force transfer functions)
    forceRAOamp, forceRAOphase = set_force_raos(vessel_params, freqs, interpolate)
    
    return WaveLoad(
        N=N,
        amp=amp,
        freqs=freqs,
        eps=eps,
        angles=angles,
        depth=depth,
        qtf_angles=qtf_angles_out,
        qtf_freqs=qtf_freqs,
        g=g,
        k=k,
        rho=rho,
        W=W,
        P=P,
        Q=Q,
        forceRAOamp=forceRAOamp,
        forceRAOphase=forceRAOphase
    )

# -------------------------------------------------------------------
# The remaining functions remain essentially unchanged except that
# I am using the attribute access (e.g. wl.angles) instead of dictionary indexing.
def set_force_raos(params, freqs, interpolate=True):
    amp_array   = jnp.asarray(params["forceRAO"]["amp"])[:, :, :, 0]
    phase_array = jnp.deg2rad(jnp.asarray(params["forceRAO"]["phase"])[:, :, :, 0])
    vessel_freqs = jnp.asarray(params["freqs"])

    if interpolate:
        def interp_over_freqs(data, left_fill_func, right_fill_func):
            angle_indices = jnp.arange(data.shape[-1])
            def interp_dof(d):
                def interp_angle(h):
                    left_val = left_fill_func(d, h)
                    right_val = right_fill_func(d, h)
                    return jnp.interp(freqs, vessel_freqs, d[:, h],
                                      left=left_val, right=right_val)
                return jax.vmap(interp_angle)(angle_indices).T
            return jax.vmap(interp_dof)(data)

        amp_abs = jnp.abs(amp_array)
        forceRAOamp = interp_over_freqs(amp_abs,
                                        left_fill_func=lambda d, h: d[0, h],
                                        right_fill_func=lambda d, h: 0.0)
        forceRAOphase = interp_over_freqs(phase_array,
                                          left_fill_func=lambda d, h: 0.0,
                                          right_fill_func=lambda d, h: d[-1, h])
    else:
        freq_indx = jnp.array([jnp.argmin(jnp.abs(vessel_freqs - w)) for w in freqs])
        n_angles = len(params["headings"])
        n_freqs_new = freqs.shape[0]
        forceRAOamp = jnp.zeros((6, n_freqs_new, n_angles))
        forceRAOphase = jnp.zeros((6, n_freqs_new, n_angles))
        for dof in range(6):
            forceRAOamp = forceRAOamp.at[dof].set(amp_array[dof, freq_indx, :])
            forceRAOphase = forceRAOphase.at[dof].set(phase_array[dof, freq_indx, :])
    return forceRAOamp, forceRAOphase

def wave_number(omega, depth, g=9.81, tol=1e-5):
    def compute_k(w):
        k_old = w**2 / g
        k_new = w**2 / (g * jnp.tanh(k_old * depth))
        diff  = jnp.abs(k_old - k_new)
        def body_fn(val):
            k_old, k_new, diff, count = val
            k_old = k_new
            k_new = w**2 / (g * jnp.tanh(k_old * depth))
            diff  = jnp.abs(k_old - k_new)
            count += 1
            return (k_old, k_new, diff, count)
        def cond_fn(val):
            return val[2] > tol
        _, k_final, _, _ = lax.while_loop(cond_fn, body_fn, (k_old, k_new, diff, 0))
        return k_final
    return jax.vmap(compute_k)(omega)

def full_qtf_6dof(qtf_headings, qtf_freqs, qtfs, wave_freqs,
                  method="Newman", interpolate=True, qtf_interp_angles=True):
    print("Generate QTF matrices".center(100, '*'))
    print(f"Using {'Newman' if method=='Newman' else 'Geometric mean'}\n")
    freq_indices = jnp.array([jnp.argmin(jnp.abs(qtf_freqs - freq)) for freq in wave_freqs])
    
    if interpolate:
        if wave_freqs[0] < qtf_freqs[0]:
            qtf_freqs = jnp.concatenate([jnp.array([0]), qtf_freqs])
            qtfs = jnp.concatenate([jnp.zeros_like(qtfs[:, :1]), qtfs], axis=1)
        def interp_freq(qtfs_dof, orig_freqs, new_freqs):
            angle_count = qtfs_dof.shape[1]
            def single_angle(h):
                data = qtfs_dof[:, h]
                return jnp.interp(new_freqs, orig_freqs, data, left=data[0], right=0.0)
            return jax.vmap(single_angle)(jnp.arange(angle_count)).T
        Qdiag = jax.vmap(interp_freq, in_axes=(0, None, None))(qtfs, qtf_freqs, wave_freqs)
        if qtf_interp_angles:
            angles_1deg = jnp.linspace(0, 2*jnp.pi, 360)
            def interp_angle(Qdiag_dof, orig_angles):
                return jax.vmap(lambda f: jnp.interp(
                    angles_1deg, orig_angles, Qdiag_dof[f, :],
                    left=Qdiag_dof[f, 0], right=Qdiag_dof[f, -1]), in_axes=0)(jnp.arange(Qdiag_dof.shape[0]))
            Qdiag = jax.vmap(interp_angle, in_axes=(0, None))(Qdiag, qtf_headings)
            qtf_angles_out = angles_1deg
        else:
            qtf_angles_out = qtf_headings
        Q = jnp.zeros((6, len(qtf_angles_out), wave_freqs.shape[0], wave_freqs.shape[0]))
        for dof in range(6):
            Qdiag_dof = Qdiag[dof]
            Qdiag_dof = jnp.swapaxes(Qdiag_dof, 0, 1)
            if method == "Newman":
                Q_dof = 0.5 * (Qdiag_dof[:, :, None] + Qdiag_dof[:, None, :])
            elif method == "geo-mean":
                sign = jnp.sign(Qdiag_dof)
                cond = jnp.sign(Qdiag_dof[:, :, None]) == jnp.sign(Qdiag_dof[:, None, :])
                Q_dof = cond * sign[:, :, None] * jnp.sqrt(jnp.abs(Qdiag_dof[:, :, None] *
                                                                    Qdiag_dof[:, None, :]))
            else:
                raise ValueError(f"Invalid method: {method}")
            Q = Q.at[dof].set(Q_dof)
    else:
        Q = jnp.zeros((6, len(qtf_headings), wave_freqs.shape[0], wave_freqs.shape[0]))
        for dof in range(6):
            Qdiag = qtfs[dof, freq_indices, :]
            if method == "Newman":
                Q_dof = 0.5 * (Qdiag[:, :, None] + Qdiag[:, None, :])
            elif method == "geo-mean":
                sign = jnp.sign(Qdiag)
                cond = jnp.sign(Qdiag[:, :, None]) == jnp.sign(Qdiag[:, None, :])
                Q_dof = cond * sign[:, :, None] * jnp.sqrt(jnp.abs(Qdiag[:, :, None] * Qdiag[:, None, :]))
            else:
                raise ValueError(f"Invalid method: {method}")
            Q = Q.at[dof].set(Q_dof)
        qtf_angles_out = qtf_headings
    # Adjust: swap yaw (index 5) with surge/sway (index 2) and zero out index 2
    Q = Q.at[5].set(Q[2])
    Q = Q.at[2].set(jnp.zeros_like(Q[2]))
    print("QTF matrices complete.".center(100, '*'))
    return Q, qtf_angles_out

def relative_incident_angle(angles, heading):
    rel_angle = angles - heading
    rel_angle = pipi(rel_angle)
    return to_positive_angle(rel_angle)

def rao_interp(forceRAOamp, forceRAOphase, qtf_angles, rel_angle):
    rel_angle_deg = jnp.rad2deg(rel_angle)
    floor_deg = (jnp.floor(rel_angle_deg / 10) * 10) % 360
    qtf_deg = jnp.rad2deg(qtf_angles)
    
    index_lb = jnp.argmin(jnp.abs(qtf_deg - floor_deg[:, None]), axis=1)
    index_ub = (index_lb + 1) % len(qtf_angles)
    freq_ind = jnp.arange(forceRAOamp.shape[1])
    
    rao_amp_lb   = forceRAOamp[:, freq_ind, index_lb]
    rao_phase_lb = forceRAOphase[:, freq_ind, index_lb]
    rao_amp_ub   = forceRAOamp[:, freq_ind, index_ub]
    rao_phase_ub = forceRAOphase[:, freq_ind, index_ub]
    
    theta1 = qtf_angles[index_lb]
    theta2 = qtf_angles[index_ub]
    scale = pipi(rel_angle - theta1) / pipi(theta2 - theta1)
    
    rao_amp   = rao_amp_lb + (rao_amp_ub - rao_amp_lb) * scale
    rao_phase = rao_phase_lb + (rao_phase_ub - rao_phase_lb) * scale
    return rao_amp, rao_phase

def first_order_loads(t, eta, wl: WaveLoad):
    rel_angle = relative_incident_angle(wl.angles, eta[-1])
    rao_amp, rao_phase = rao_interp(wl.forceRAOamp, wl.forceRAOphase, wl.qtf_angles, rel_angle)
    x = eta[0]
    y = eta[1]
    rao = rao_amp * jnp.cos(
        wl.freqs * t - wl.k * x * jnp.cos(wl.angles) - wl.k * y * jnp.sin(wl.angles)
        - wl.eps - rao_phase)
    tau_wf = jnp.dot(rao, wl.amp)
    return tau_wf

def second_order_loads(t, heading, wl: WaveLoad):
    rel_angle = jnp.mean(relative_incident_angle(wl.angles, heading))
    angles_1deg = jnp.linspace(0, 2*jnp.pi, 360)
    heading_index = jnp.argmin(jnp.abs(angles_1deg - rel_angle))
    
    Q = wl.Q[:, heading_index, :, :]
    exp_term = jnp.exp(wl.W * (1j * t) - 1j * wl.P)
    
    tau_sv = jnp.zeros(6)
    for dof in range(6):
        Q_dof = Q[dof]
        term = Q_dof * exp_term
        temp = jnp.dot(term, wl.amp)
        tau_sv_dof = jnp.real(jnp.dot(wl.amp, temp))
        tau_sv = tau_sv.at[dof].set(tau_sv_dof)
    return tau_sv

def wave_load(t, eta, wl: WaveLoad):
    return first_order_loads(t, eta, wl) + second_order_loads(t, eta[-1], wl)
