"""
Adaptive controller with Fourier series-based disturbance model for marine vessel dynamics.
This script implements a Model Reference Adaptive Control (MRAC) framework using a backstepping
design to counteract disturbances (e.g., wave-induced forces) modeled via a truncated Fourier series.
Key functionalities include:
  - Converting full 6-DOF vessel dynamics to a practical 3-DOF representation.
  - Generating an exact set of frequency components using np.linspace.
  - Constructing a (2N+1)-dimensional Fourier regressor:
        [1, cos(w₁·t), sin(w₁·t), ..., cos(w_N·t), sin(w_N·t)]
  - Updating adaptive parameters via an adaptive law with parameter projection to maintain bounded estimates.
  - Computing the control input tau using a combination of feedback (with gains K1 and K2),
    vessel dynamics (M and D), and adaptive compensation from the disturbance model.

The control law is given by:
    tau = -K2 · z₂ + D · α + M · α̇ - Φᵀ · θ̂
where:
    • z₁ = Rᵀ · (η - η_d)         (transformed position error)
    • α   = Rᵀ · η̇_d - (K1+k) · z₁   (virtual control input)
    • z₂ = ν - α                  (velocity error)
    • Φ is the block-diagonal Fourier regressor matrix
    • θ̂ is the adaptive parameter vector updated by:
          θ̂_dot = γ · (Φ · z₂)

Author: Kristian Magnus Roen (Orginial author Theodor Brørby)
Date: 12.02.2025
"""


import numpy as np
from mclsimpy.utils import Rz, six2threeDOF, three2sixDOF, pipi, Smat

class AdaptiveFSController:
    """
    Adaptive controller using a truncated Fourier series-based disturbance model
    in a Model Reference Adaptive Control (MRAC) framework. The controller uses
    a backstepping design that incorporates adaptive update laws to counteract
    disturbances (e.g., waves) modeled as a Fourier series.

    Improvements in this version include:
      - Exact frequency generation using np.linspace.
      - Type hints and improved documentation.
      - More practical default tuning parameters.
      - Parameter projection to ensure boundedness of adaptive parameters.
    """

    def __init__(self, dt: float, M: np.ndarray, D: np.ndarray, N: int = 15,
                 K1: np.ndarray = None, K2: np.ndarray = None, gamma: np.ndarray = None,
                 kappa: float = 1.0, theta_bound: float = 1e3):
        """
        Initialize the AdaptiveFSController.

        Parameters
        ----------
        dt : float
            Timestep (seconds).
        M : np.ndarray
            Mass matrix (6×6) of the vessel (inertia plus added mass).
        D : np.ndarray
            Full damping matrix (6×6) of the vessel.
        N : int, optional
            Number of frequency components. The Fourier series regressor will have
            a dimension of (2N+1). Default is 15.
        K1 : np.ndarray, optional
            Feedback gain for the first error dynamics (3×3). If None, defaults to
            a diagonal matrix.
        K2 : np.ndarray, optional
            Feedback gain for the second error dynamics (3×3). If None, defaults to
            a diagonal matrix.
        gamma : np.ndarray, optional
            Adaptive gain matrix. If None, defaults to a diagonal matrix of size
            (3*(2N+1)×3*(2N+1)).
        kappa : float, optional
            A positive scalar used in the virtual control design.
        theta_bound : float, optional
            Bound used in parameter projection (absolute value for each parameter).
        """
        if dt <= 0:
            raise ValueError("Timestep dt must be positive.")
        self._dt = dt

        # Convert the provided matrices from 6-DOF to 3-DOF representations.
        self._M = six2threeDOF(M)
        self._D = six2threeDOF(D)

        # Setup frequency components.
        self._N = N
        # Default frequency bounds (in rad/s):
        w_min = 2 * np.pi / 20
        w_max = 2 * np.pi / 2
        self.set_freqs(w_min, w_max, N)

        # Initialize the adaptive parameter vector (dimension: 3*(2N+1)).
        self.theta_hat = np.zeros((2 * self._N + 1) * 3)

        # Tuning gains. Defaults are more moderate; tune them as necessary.
        if K1 is None:
            self._K1 = np.diag([1, 1, 1])*0.1
        else:
            self._K1 = np.array(K1)

        if K2 is None:
            self._K2 = np.diag([0.5, 0.5, 0.5])*0.1
        else:
            self._K2 = np.array(K2)

        if gamma is None:
            self._gamma = np.eye((2 * self._N + 1) * 3) * 0.4
        else:
            gamma = np.array(gamma)
            if gamma.shape != ((2 * self._N + 1) * 3, (2 * self._N + 1) * 3):
                raise ValueError("Adaptive gain gamma must be of shape (3*(2N+1), 3*(2N+1)).")
            self._gamma = gamma

        if kappa <= 0:
            raise ValueError("kappa must be positive.")
        self._kappa = kappa

        # Parameter projection bound.
        self._theta_bound = theta_bound

    def set_freqs(self, w_min: float, w_max: float, N: int):
        """
        Set the frequency components for the Fourier series disturbance model.

        Parameters
        ----------
        w_min : float
            Lower bound frequency (in rad/s).
        w_max : float
            Upper bound frequency (in rad/s).
        N : int
            Number of frequency components.
        """
        self._N = N
        # Use np.linspace to guarantee exactly N frequencies.
        self._freqs = np.linspace(w_min, w_max, N)

    def get_regressor(self, t: float) -> np.ndarray:
        """
        Generate the (2N+1)-dimensional regressor vector based on time t.

        The regressor is defined as:
            [1, cos(w1*t), sin(w1*t), cos(w2*t), sin(w2*t), ..., cos(wN*t), sin(wN*t)].

        Parameters
        ----------
        t : float
            Time (seconds).

        Returns
        -------
        np.ndarray
            Regressor vector of shape ((2N+1),).
        """
        regressor = np.empty(2 * self._N + 1)
        regressor[0] = 1.0
        for i in range(self._N):
            regressor[2 * i + 1] = np.cos(self._freqs[i] * t)
            regressor[2 * i + 2] = np.sin(self._freqs[i] * t)
        return regressor

    def get_tau(self, eta: np.ndarray, eta_d: np.ndarray, nu: np.ndarray,
                eta_d_dot: np.ndarray, eta_d_ddot: np.ndarray, t: float,
                calculate_bias: bool = False):
        """
        Compute the control input tau and update the adaptive parameters.

        Parameters
        ----------
        eta : np.ndarray
            Current vessel configuration in NED (3 DOF: surge, sway, yaw).
        eta_d : np.ndarray
            Desired vessel configuration in NED (3 DOF).
        nu : np.ndarray
            Current vessel body-frame velocities [u, v, r] (3 DOF).
        eta_d_dot : np.ndarray
            Desired velocity in NED (3 DOF).
        eta_d_ddot : np.ndarray
            Desired acceleration in NED (3 DOF).
        t : float
            Current time (seconds).
        calculate_bias : bool, optional
            If True, returns a diagnostic dictionary containing the estimated bias and
            intermediate control terms.

        Returns
        -------
        tau : np.ndarray
            Computed control forces/torques (3 DOF).
        debug : dict, optional
            Diagnostic information (only returned if calculate_bias is True).
        """
        # Compute rotation matrix based on yaw.
        R = Rz(eta[-1])
        # Skew-symmetric matrix corresponding to yaw rate.
        S = Smat(np.array([0, 0, nu[-1]]))

        # Compute the regressor vector.
        regressor = self.get_regressor(t)
        # Build block-diagonal regressor Phi (each block corresponds to surge, sway, and yaw).
        Phi = np.block([
            [regressor.reshape(-1, 1), np.zeros((2 * self._N + 1, 2))],
            [np.zeros((2 * self._N + 1, 1)), regressor.reshape(-1, 1), np.zeros((2 * self._N + 1, 1))],
            [np.zeros((2 * self._N + 1, 2)), regressor.reshape(-1, 1)]
        ])

        # Define the transformed position error.
        z1 = R.T @ (eta - eta_d)
        z1[-1] = pipi(z1[-1])

        # Virtual control design.
        alpha0 = -self._kappa * z1
        alpha = R.T @ eta_d_dot - self._K1 @ z1 + alpha0

        # Velocity error.
        z2 = nu - alpha

        # Compute derivative of z1.
        z1_dot = -S @ z1 + z2 - (self._K1 + self._kappa * np.eye(3)) @ z1

        # Compute derivative of alpha.
        # This follows the original design with an extra term for the derivative of R.T:
        alpha_dot = -(self._K1 + self._kappa * np.eye(3)) @ z1_dot - S @ (R.T @ eta_d_dot) + R.T @ eta_d_ddot

        # Adaptive update law.
        theta_hat_dot = self._gamma @ (Phi @ z2)
        self.theta_hat += theta_hat_dot * self._dt

        # Parameter projection to keep theta_hat bounded.
        self.theta_hat = np.clip(self.theta_hat, -self._theta_bound, self._theta_bound)

        # Control law.
        tau = -self._K2 @ z2 + self._D @ alpha + self._M @ alpha_dot - Phi.T @ self.theta_hat

        if calculate_bias:
            b_hat = Phi.T @ self.theta_hat
            debug = {
                "b_hat": b_hat,
                "tau_z2": -self._K2 @ z2,
                "tau_alpha": self._D @ alpha,
                "tau_alpha_dot": self._M @ alpha_dot,
                "z1": z1,
                "z2": z2,
            }
            return tau, debug

        return tau

    def set_tuning_params(self, K1: list, K2: list, gamma: list):
        """
        Set new tuning parameters for the controller.

        Parameters
        ----------
        K1 : list
            List of diagonal entries for the K1 matrix (length 3).
        K2 : list
            List of diagonal entries for the K2 matrix (length 3).
        gamma : list
            List of diagonal entries for the gamma matrix. Must have length equal to 3*(2N+1).
        """
        self._K1 = np.diag(K1)
        self._K2 = np.diag(K2)
        if len(gamma) != (2 * self._N + 1) * 3:
            raise ValueError("Length of gamma must be equal to 3*(2N+1).")
        self._gamma = np.diag(gamma)

    def get_theta(self) -> np.ndarray:
        """
        Get the current adaptive parameter estimates.

        Returns
        -------
        np.ndarray
            A copy of the current theta_hat vector.
        """
        return self.theta_hat.copy()