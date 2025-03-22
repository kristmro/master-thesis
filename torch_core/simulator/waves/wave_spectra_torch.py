#!/usr/bin/env python3
"""
wave_spectra_torch.py

Differentiable wave spectra module implemented with PyTorch.
This module replicates the functionality of the old wave_spectra.py classes
(BaseSpectrum, BasePMSpectrum, ModifiedPiersonMoskowitz, and JONSWAP)
but uses torch operations for full differentiability.

Author: [Kristian Magnus Roen/ adapted from Jan-Erik Hygen]
Date:   2025-02-17
"""

import torch
import math
from abc import ABC, abstractmethod


class BaseSpectrum(ABC):
    """
    Base class for 1-D Wave Spectra implemented in PyTorch.

    Parameters
    ----------
    freq : 1D array-like or Tensor
        Frequencies of the wave spectrum.
    freq_hz : bool
        If True, frequencies are given in Hz and converted to rad/s.
    """
    def __init__(self, freq, freq_hz=False):
        # Ensure freq is a torch float tensor
        freq_t = torch.tensor(freq, dtype=torch.float32)
        if freq_hz:
            freq_t = freq_t * (2.0 * math.pi)  # convert Hz -> rad/s
        self._freq = freq_t
        self._freq_hz = freq_hz

    def __call__(self, *args, freq_hz=None, **kwargs):
        """
        Evaluate the spectrum at self._freq.

        Returns
        -------
        freq : Tensor
            The (angular) frequencies used, shape (N,).
        spectrum : Tensor
            The spectral density at each freq, shape (N,).
        """
        freq = self._freq.clone()
        # freq_hz param is available for consistency with old code,
        # but is not strictly needed if we've already converted in constructor.
        spectrum = self._spectrum(freq, *args, **kwargs)
        return freq, spectrum

    def moment(self, n, *args, **kwargs):
        """
        Calculate the n-th spectral moment:
           m_n = ∫(ω^n * S(ω)) dω

        This uses torch.trapz for numerical integration.

        Parameters
        ----------
        n : int
            The exponent for the frequency in the integrand.

        Returns
        -------
        m_n : Tensor (scalar)
            The n-th moment of the spectrum.
        """
        freq, spec = self.__call__(*args, **kwargs)
        # trapezoidal rule in PyTorch
        return torch.trapz((freq**n) * spec, freq)

    def realization(self, time, *args, **kwargs):
        """
        Generate a wave realization at a fixed point (x=0) from this spectrum.

        Summation of cosines:
          η(t) = Σ sqrt(2*S(ω_i)*Δω) * cos(ω_i * t + φ_i)

        Parameters
        ----------
        time : 1D array-like or Tensor
            Time points at which to evaluate the wave elevation.

        Returns
        -------
        timeseries : Tensor of shape (len(time),)
            The wave elevation as a function of time.
        """
        freq, spectrum = self.__call__(*args, **kwargs)
        # Assume uniform freq spacing to define Δω
        dw = freq[1] - freq[0]
        # amplitude array
        amp = torch.sqrt(2.0 * spectrum * dw)

        # random phases in [0, 2π)
        eps = 2.0 * math.pi * torch.rand(len(amp), dtype=amp.dtype, device=amp.device)

        # Expand time and freq for broadcast
        # time shape => (T,)
        time_t = torch.tensor(time, dtype=torch.float32, device=amp.device).unsqueeze(-1)  # (T,1)
        freq_t = freq.unsqueeze(0)  # (1,N)
        eps_t  = eps.unsqueeze(0)   # (1,N)

        # argument = ω * t + φ
        # shape => (T,N)
        argument = freq_t * time_t + eps_t

        # sum over freq dimension => shape(T,)
        eta_t = torch.sum(amp * torch.cos(argument), dim=1)
        return eta_t

    @abstractmethod
    def _spectrum(self, omega, *args, **kwargs):
        """
        Compute the spectral density at each omega in 'omega'.

        Must be overridden by child classes.
        """
        raise NotImplementedError


class BasePMSpectrum(BaseSpectrum):
    """
    Base class for Pierson-Moskowitz-type spectra.

    The old code did:
      spectrum = A / ω^5 * exp(-B / ω^4)
    """
    def __call__(self, A, B, freq_hz=None):
        return super().__call__(A, B, freq_hz=freq_hz)

    def _spectrum(self, omega, A, B):
        # Avoid dividing by zero at ω=0
        # or at very small frequencies. The original code doesn't handle that,
        # but we can clamp the frequency to avoid nans in derivative-based computations if needed.
        # For now, replicate original logic exactly.
        return A / (omega**5) * torch.exp(-B / (omega**4))


class ModifiedPiersonMoskowitz(BasePMSpectrum):
    """
    The "Modified Pierson-Moskowitz" wave spectrum.

    The old formula:
        A = (5/16) * Hs^2 * ω_p^4
        B = (5/4) * ω_p^4
    Where ω_p = 2π / T_p.

    Then calls the parent BasePMSpectrum with (A,B).
    """
    def __call__(self, hs, tp, freq_hz=None):
        """
        Evaluate the MPM spectrum.

        Parameters
        ----------
        hs : float
            Significant wave height [m].
        tp : float
            Peak period [s].
        freq_hz : bool, optional
            If True, might convert freq in constructor. (Included for backward compat.)

        Returns
        -------
        freq : Tensor of shape (N,)
        spectrum : Tensor of shape (N,)
        """
        A = self._A(hs, tp)
        B = self._B(tp)
        return super().__call__(A, B, freq_hz=freq_hz)

    def _A(self, hs, tp):
        wp = 2.0 * math.pi / tp
        return (5.0 / 16.0) * (hs**2) * (wp**4)

    def _B(self, tp):
        wp = 2.0 * math.pi / tp
        return (5.0 / 4.0) * (wp**4)


class JONSWAP(ModifiedPiersonMoskowitz):
    """
    JONSWAP wave spectrum, building on MPM, with additional γ factor:

        S_J(ω) = α * S_PM(ω) * γ^(b(ω)),
    where
       α = 1 - 0.287 ln(γ),
       b(ω) = exp[-0.5 ((ω - ω_p)/(σ ω_p))^2],
    and
       σ = {0.07  if ω <= ω_p,
            0.09  if ω > ω_p}.

    The default gamma is often 3.3, but can be set any float.
    """
    def __call__(self, hs, tp, gamma=3.3, freq_hz=None):
        """
        Evaluate the JONSWAP spectrum.

        Parameters
        ----------
        hs : float
            Significant wave height.
        tp : float
            Peak period.
        gamma : float, default=3.3
            Peak enhancement factor.
        freq_hz : bool, optional

        Returns
        -------
        freq : Tensor (N,)
        spectrum : Tensor (N,)
        """
        freq, pm_spectrum = super().__call__(hs, tp, freq_hz=freq_hz)

        alpha = self._alpha(gamma)
        b = self._b(tp, freq)

        # final = α * PM * (γ^b)
        spec_jonswap = alpha * pm_spectrum * (gamma**b)
        return freq, spec_jonswap

    def _alpha(self, gamma):
        # alpha = 1 - 0.287 ln(γ)
        return 1.0 - 0.287 * torch.log(torch.tensor(gamma, dtype=torch.float32))

    def _b(self, tp, freq):
        # b(ω) = exp[-0.5 ((ω - ω_p)/(σ ω_p))^2]
        wp = 2.0 * math.pi / tp
        sigma = self._sigma(wp, freq)
        return torch.exp(-0.5 * ((freq - wp) / (sigma * wp))**2)

    def _sigma(self, wp, freq):
        """
        piecewise: σ = 0.07 if ω <= ω_p, else 0.09
        """
        sigma = torch.empty_like(freq)
        mask = (freq <= wp)
        sigma[mask]  = 0.07
        sigma[~mask] = 0.09
        return sigma