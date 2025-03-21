"""
utils.py

Author: Kristian Magnus Roen
Date:   2025-03-17
"""

import jax.numpy as jnp
import numpy as np
import jax
from functools import partial
from jax.scipy.linalg import block_diag
# ---------------------------------------------------------------------------
# 1) Helpers for angles, "to_positive_angle", "pipi", etc.
#    You can keep your original Python definitions but replace np with jnp:
# ---------------------------------------------------------------------------
def to_positive_angle(angle):
    """
    Force angle into [0, 2*pi).
    """
    return jnp.where(angle < 0, angle + 2*jnp.pi, angle)

def pipi(angle):
    """
    Force angle into [-pi, pi).
    """
    return jnp.mod(angle + jnp.pi, 2*jnp.pi) - jnp.pi

@jax.jit
def three2sixDOF(v):
    """
    Convert a 3DOF vector or matrix to 6DOF.
    
    - If `v` is a vector of shape (3,), returns an array of shape (6,) with:
      [v[0], v[1], 0, 0, 0, v[2]].
    
    - If `v` is a matrix of shape (3, 3), it flattens `v` and embeds it into a 6x6 matrix.
      The embedding is defined by:
          part1 = flat[0:2]
          part2 = zeros(3)
          part3 = flat[2:5]
          part4 = zeros(3)
          part5 = flat[5:6]
          part6 = zeros(18)
          part7 = flat[6:8]
          part8 = zeros(3)
          part9 = flat[8:9]
      The concatenated vector of length 36 is then reshaped to (6,6).
    """
    if v.ndim == 1:
        if v.shape[0] != 3:
            raise ValueError(f"Expected 1D array with 3 elements, got shape {v.shape}")
        return jnp.array([v[0], v[1], 0.0, 0.0, 0.0, v[2]], dtype=v.dtype)
    elif v.ndim == 2:
        if v.shape != (3, 3):
            raise ValueError(f"Expected 2D array of shape (3,3), got shape {v.shape}")
        flat = jnp.ravel(v)
        part1 = flat[0:2]
        part2 = jnp.zeros(3, dtype=v.dtype)
        part3 = flat[2:5]
        part4 = jnp.zeros(3, dtype=v.dtype)
        part5 = flat[5:6]
        part6 = jnp.zeros(18, dtype=v.dtype)
        part7 = flat[6:8]
        part8 = jnp.zeros(3, dtype=v.dtype)
        part9 = flat[8:9]
        out_flat = jnp.concatenate([part1, part2, part3, part4, part5, part6, part7, part8, part9])
        return jnp.reshape(out_flat, (6, 6))
    else:
        raise ValueError("Input v must be a 1D or 2D array.")
@jax.jit
def six2threeDOF(v):
    """
    Convert a 6DOF vector or matrix to 3DOF.
    
    - If `v` is a vector of shape (6,), returns the elements at indices [0, 1, 5] → shape (3,).
    - If `v` is a matrix of shape (T,6) with T != 6, returns the columns [0, 1, 5] → shape (T,3).
    - If `v` is a 6x6 matrix, returns the submatrix with rows and columns [0, 1, 5] → shape (3,3).
    """
    if isinstance(v, np.ndarray):
        v = jnp.array(v)
        
    if v.ndim == 1:
        if v.shape[0] != 6:
            raise ValueError(f"Expected 1D array with 6 elements, got shape {v.shape}")
        return v[jnp.array([0, 1, 5])]
    elif v.ndim == 2:
        rows, cols = v.shape
        if rows == 6 and cols == 6:
            # Extract the submatrix with rows and cols [0,1,5]
            idx = jnp.array([0, 1, 5])
            return v[idx][:, idx]
        elif cols == 6:
            return v[:, jnp.array([0, 1, 5])]
        else:
            raise ValueError(f"Expected shape (T,6) or (6,6), got {v.shape} instead.")
    else:
        raise ValueError(f"Input must be a 1D or 2D array, but got array with {v.ndim} dimensions.")
# --------------------------------------------------------------------------
# Rotational matrix functions
# --------------------------------------------------------------------------
def Rx(phi):
    """Return the 3x3 rotation matrix about the x-axis by angle phi."""
    c = jnp.cos(phi)
    s = jnp.sin(phi)
    return jnp.array([[1,  0,  0],
                      [0,  c, -s],
                      [0,  s,  c]], dtype=jnp.float32)

def Ry(theta):
    """Return the 3x3 rotation matrix about the y-axis by angle theta."""
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return jnp.array([[ c, 0, s],
                      [ 0, 1, 0],
                      [-s, 0, c]], dtype=jnp.float32)

def Rz(psi):
    """Return the 3x3 rotation matrix about the z-axis by angle psi."""
    c = jnp.cos(psi)
    s = jnp.sin(psi)
    return jnp.array([[c, -s, 0],
                      [s,  c, 0],
                      [0,  0, 1]], dtype=jnp.float32)

def Rzyx(eta):
    """ Full roation matrix"""
    phi, theta, psi = eta[3], eta[4], eta[5]
    return Rz(psi) @ Ry(theta) @ Rx(phi)

def Tzyx(eta):
    phi, theta, psi = eta[3], eta[4], eta[5]
    return jnp.array([
        [1, jnp.sin(phi)*jnp.tan(theta), jnp.cos(phi)*jnp.tan(theta)],
        [0, jnp.cos(phi), -jnp.sin(phi)],
        [0, jnp.sin(phi)/jnp.cos(theta), jnp.cos(phi)/jnp.cos(theta)]
    ])

def J(eta):
    """6 DOF rotation matrix."""

    roll, pitch, yaw = eta[3], eta[4], eta[5]
    return jnp.block([
        [Rzyx(eta), jnp.zeros((3, 3))],
        [jnp.zeros((3, 3)), Tzyx(eta)]
    ])



def rk4_step_impl(x, dt, f, *args):
    k1 = f(x, *args)
    k2 = f(x + 0.5 * dt * k1, *args)
    k3 = f(x + 0.5 * dt * k2, *args)
    k4 = f(x + dt * k3, *args)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

rk4_step = jax.jit(rk4_step_impl, static_argnums=(2,))

def rk38_step(x, dt, f, *args):
    """
    Performs one RK3/8 integration step.
    
    Parameters:
        x: current state (JAX array)
        dt: time step (float)
        f: function that computes the derivative, with signature f(x, *args)
        *args: additional parameters passed to f
    
    Returns:
        x_next: state after time dt
    """
    k1 = f(x, *args)
    k2 = f(x + (dt/3.0) * k1, *args)
    k3 = f(x + dt * (-1.0/3.0 * k1 + 1.0 * k2), *args)
    k4 = f(x + dt * (k1 - k2 + k3), *args)
    return x + dt * (1.0/8.0 * k1 + 3.0/8.0 * k2 + 3.0/8.0 * k3 + 1.0/8.0 * k4)


def random_ragged_spline(key, T_total, num_knots, poly_orders, deriv_orders,
                         min_step, max_step, min_knot, max_knot):
    """TODO: docstring."""
    poly_orders = np.array(poly_orders).ravel().astype(int)
    deriv_orders = np.array(deriv_orders).ravel().astype(int)
    num_dims = poly_orders.size
    assert deriv_orders.size == num_dims
    shape = (num_dims,)
    knots = uniform_random_walk(key, num_knots - 1, shape, min_step, max_step)
    knots = jnp.clip(knots, min_knot, max_knot)
    flat_knots = jnp.reshape(knots, (num_knots, -1))
    diffs = jnp.linalg.norm(jnp.diff(flat_knots, axis=0), axis=1)
    T = T_total * (diffs / jnp.sum(diffs))
    t_knots = jnp.concatenate((jnp.array([0., ]),
                               jnp.cumsum(T))).at[-1].set(T_total)
    coefs = []
    for i, (p, d) in enumerate(zip(poly_orders, deriv_orders)):
        coefs.append(smooth_trajectory(knots[:, i], t_knots, p, d))
    coefs = tuple(coefs)
    knots = tuple(knots[:, i] for i in range(num_dims))
    return t_knots, knots, coefs

def uniform_random_walk(key, num_steps, shape=(), min_step=0., max_step=1.):
    """TODO: docstring."""
    minvals = jnp.broadcast_to(min_step, shape)
    maxvals = jnp.broadcast_to(max_step, shape)
    noise = minvals + (maxvals - minvals)*jax.random.uniform(key, (num_steps,
                                                                   *shape))
    points = jnp.concatenate((jnp.zeros((1, *shape)),
                              jnp.cumsum(noise, axis=0)), axis=0)
    return points

@partial(jax.jit, static_argnums=(2, 3))
def smooth_trajectory(x_knots, t_knots, poly_order, deriv_order):
    """TODO: docstring."""
    # TODO: shape checking
    num_knots = x_knots.shape[0]
    knot_shape = x_knots.shape[1:]
    flat_x_knots = jnp.reshape(x_knots, (num_knots, -1))
    in_axes = (1, None, None, None)
    out_axes = (2, 1, 1)
    flat_coefs, _, _ = jax.vmap(_scalar_smooth_trajectory,
                                in_axes, out_axes)(flat_x_knots, t_knots,
                                                   poly_order, deriv_order)
    num_polys = num_knots - 1
    coefs = jnp.reshape(flat_coefs, (num_polys, poly_order + 1, *knot_shape))
    return coefs

@partial(jax.jit, static_argnums=(2, 3))
def _scalar_smooth_trajectory(x_knots, t_knots, poly_order, deriv_order):
    """Construct a smooth trajectory through given points.

    Arguments
    ---------
    x_knots : jax.numpy.ndarray
        TODO.
    t_knots : jax.numpy.ndarray
        TODO.
    poly_order : int
        TODO.
    deriv_order : int
        TODO.

    Returns
    -------
    coefs : jax.numpy.ndarray
        TODO.

    References
    ----------
    .. [1] Charles Richter, Adam Bry, and Nicholas Roy,
           "Polynomial trajectory planning for aggressive quadrotor flight in
           dense indoor environments", ISRR 2013.
    .. [2] Daniel Mellinger and Vijay Kumar,
           "Minimum snap trajectory generation and control for quadrotors",
           ICRA 2011.
    .. [3] Declan Burke, Airlie Chapman, and Iman Shames,
           "Generating minimum-snap quadrotor trajectories really fast",
           IROS 2020.
    """
    num_coefs = poly_order + 1          # number of coefficients per polynomial
    num_knots = x_knots.size            # number of interpolating points
    num_polys = num_knots - 1           # number of polynomials
    primal_dim = num_coefs * num_polys  # number of unknown coefficients

    T = jnp.diff(t_knots)                # polynomial lengths in time
    powers = jnp.arange(poly_order + 1)  # exponents defining each monomial
    D = jnp.diag(powers[1:], -1)         # maps monomials to their derivatives

    c0 = jnp.zeros((deriv_order + 1, num_coefs)).at[0, 0].set(1.)
    c1 = jnp.zeros((deriv_order + 1, num_coefs)).at[0, :].set(1.)
    for n in range(1, deriv_order + 1):
        c0 = c0.at[n].set(D @ c0[n - 1])
        c1 = c1.at[n].set(D @ c1[n - 1])

    # Assemble constraints in the form `A @ x = b`, where `x` is the vector of
    # stacked polynomial coefficients

    # Knots
    b_knots = jnp.concatenate((x_knots[:-1], x_knots[1:]))
    A_knots = jnp.vstack([
        block_diag(*jnp.tile(c0[0], (num_polys, 1))),
        block_diag(*jnp.tile(c1[0], (num_polys, 1)))
    ])

    # Zero initial conditions (velocity, acceleration, jerk)
    b_init = jnp.zeros(deriv_order - 1)
    A_init = jnp.zeros((deriv_order - 1, primal_dim))
    A_init = A_init.at[:deriv_order - 1, :num_coefs].set(c0[1:deriv_order])

    # Zero final conditions (velocity, acceleration, jerk)
    b_fin = jnp.zeros(deriv_order - 1)
    A_fin = jnp.zeros((deriv_order - 1, primal_dim))
    A_fin = A_fin.at[:deriv_order - 1, -num_coefs:].set(c1[1:deriv_order])

    # Continuity (velocity, acceleration, jerk, snap)
    b_cont = jnp.zeros(deriv_order * (num_polys - 1))
    As = []
    zero_pad = jnp.zeros((num_polys - 1, num_coefs))
    Tn = jnp.ones_like(T)
    for n in range(1, deriv_order + 1):
        Tn = T * Tn
        diag_c0 = block_diag(*(c0[n] / Tn[1:].reshape([-1, 1])))
        diag_c1 = block_diag(*(c1[n] / Tn[:-1].reshape([-1, 1])))
        As.append(jnp.hstack((diag_c1, zero_pad))
                  - jnp.hstack((zero_pad, diag_c0)))
    A_cont = jnp.vstack(As)

    # Assemble
    A = jnp.vstack((A_knots, A_init, A_fin, A_cont))
    b = jnp.concatenate((b_knots, b_init, b_fin, b_cont))
    dual_dim = b.size

    # Compute the cost Hessian `Q(T)` as a function of the length `T` for each
    # polynomial, and stack them into the full block-diagonal Hessian
    ij_1 = powers.reshape([-1, 1]) + powers + 1
    D_snap = jnp.linalg.matrix_power(D, deriv_order)
    Q_snap = D_snap @ (1 / ij_1) @ D_snap.T
    Q_poly = lambda T: Q_snap / (T**(2*deriv_order - 1))  # noqa: E731
    Q = block_diag(*jax.vmap(Q_poly)(T))

    # Assemble KKT system and solve for coefficients
    K = jnp.block([
        [Q, A.T],
        [A, jnp.zeros((dual_dim, dual_dim))]
    ])
    soln = jnp.linalg.solve(K, jnp.concatenate((jnp.zeros(primal_dim), b)))
    primal, dual = soln[:primal_dim], soln[-dual_dim:]
    coefs = primal.reshape((num_polys, -1))
    r_primal = A@primal - b
    r_dual = Q@primal + A.T@dual
    return coefs, r_primal, r_dual


@jax.jit
def spline(t, t_knots, coefs):
    """Compute the value of a polynomial spline at time `t`."""
    num_polys = coefs.shape[0]
    poly_order = coefs.shape[1] - 1
    powers = jnp.arange(poly_order + 1)
    i = jnp.clip(jnp.searchsorted(t_knots, t, side='left') - 1,
                 0, num_polys - 1)
    tau = (t - t_knots[i]) / (t_knots[i+1] - t_knots[i])
    x = jnp.tensordot(coefs[i], tau**powers, axes=(0, 0))
    return x
