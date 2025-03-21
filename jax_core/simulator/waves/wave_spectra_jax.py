"""
wave_spectra_jax.py

Implements wave spectral models (JONSWAP, modified Pierson-Moskowitz) and associated helper functions in JAX, enabling the generation of realistic sea-state time series from spectral descriptions, essential for wave load simulations on vessels.

Author: Kristian Magnus Roen
Date:   2025-03-17
"""


import jax
import jax.numpy as jnp

# -------------------------------------------------------------
# Basic helper functions
# -------------------------------------------------------------
def base_spectrum(freq, spectrum_fn, *args, freq_hz=False, **kwargs):
    omega = jnp.array(freq)
    if freq_hz:
        omega = 2 * jnp.pi * omega
    _, S = spectrum_fn(omega, *args, **kwargs)

    return omega, S

def moment(omega, S, n):
    """
    Compute the n-th spectral moment using a simple rectangular rule.
    
    Parameters:
      omega: 1D array of angular frequencies.
      S    : Spectrum values at omega.
      n    : Order of the moment.
      
    Returns:
      m_n: The n-th moment.
    """
    
    return jnp.trapezoid(S * omega**n, omega)

def realization(key, time, spectrum_fn, freq, *args, freq_hz=False, **kwargs):
    """
    Generate a wave elevation time series from a spectrum using random phases.
    
    Parameters:
      key        : PRNG key for jax.random.
      time       : 1D array of time instants.
      spectrum_fn: Function that returns S(omega) given omega and additional parameters.
      freq       : Frequency array (in Hz if freq_hz is True or rad/s otherwise).
      *args, **kwargs: Additional parameters for spectrum_fn.
      freq_hz    : Boolean flag indicating whether freq is in Hz.
    
    Returns:
      wave: 1D array (length = len(time)) representing the wave elevation time series.
    """
    omega, S = base_spectrum(freq, spectrum_fn, *args, freq_hz=freq_hz, **kwargs)
    dw = omega[1] - omega[0]
    amp = jnp.sqrt(2.0 * S * dw)
    # Generate random phases for each frequency component
    eps = jax.random.uniform(key, shape=(len(amp),), minval=0, maxval=2*jnp.pi)
    return jnp.sum(amp * jnp.cos(omega * time[:, None] + eps), axis=1)

# -------------------------------------------------------------
# Spectrum definitions
# -------------------------------------------------------------
def pm_spectrum(omega, A, B):
    """
    Base Pierson-Moskowitz-type spectrum.
    
    S(omega) = A / omega^5 * exp(-B / omega^4)
    """
    return A / omega**5 * jnp.exp(-B / omega**4)


def modified_pierson_moskowitz_spectrum(freq, hs, tp, freq_hz=False):
    """
    Compute the Modified Pierson-Moskowitz spectrum.
    
    Parameters:
      freq   : 1D array of frequencies.
      hs     : Significant wave height.
      tp     : Peak period.
      freq_hz: If True, freq is in Hz (will be converted to rad/s).
    
    Returns:
      (omega, S): Angular frequency grid and the corresponding spectrum.
    """
    omega = jnp.array(freq)
    if freq_hz:
        omega = 2 * jnp.pi * omega
    wp = 2 * jnp.pi / tp
    A = (5.0 / 16.0) * hs**2 * wp**4
    B = (5.0 / 4.0) * wp**4
    S = pm_spectrum(omega, A, B)
    return omega, S

def jonswap_spectrum(freq, hs, tp, gamma=1.0, freq_hz=False):
    """
    Compute the JONSWAP spectrum.
    
    Based on a Modified Pierson-Moskowitz spectrum with a peakedness adjustment.
    
    Parameters:
      freq   : 1D array of frequencies.
      hs     : Significant wave height.
      tp     : Peak period.
      gamma  : Peak enhancement factor.
      freq_hz: If True, freq is in Hz (will be converted to rad/s).
    
    Returns:
      (omega, S): Angular frequency grid and the JONSWAP spectrum.
    """
    omega, pm = modified_pierson_moskowitz_spectrum(freq, hs, tp, freq_hz=freq_hz)
    alpha = 1.0 - 0.287 * jnp.log(gamma)
    sigma = jnp.where(omega <= (2 * jnp.pi / tp), 0.07, 0.09)
    wp = 2 * jnp.pi / tp
    b = jnp.exp(-0.5 * ((omega - wp) / (sigma * wp))**2)
    S = alpha * pm * gamma**b
    return omega, S

# -------------------------------------------------------------
# Example usage:
# -------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    hs = 5.0            # Significant wave height
    tp = 9.0     # Peak period
    gamma = 3.3                # JONSWAP peak factor

    wp = 2 * jnp.pi / tp       # Peak frequency
    wmin = 0.5 * wp
    wmax = 3.0 * wp
    N = 100  # Number of wave components
    wave_freqs = jnp.linspace(wmin, wmax, N)
    # Compute wave spectrum using our JONSWAP functional spectrum
    # (freq_hz=False because wave_freqs are already in rad/s)
    omega, wave_spectrum = jonswap_spectrum(wave_freqs, hs, tp, gamma=gamma, freq_hz=False)
    dw = (wmax - wmin) / N
    wave_amps = jnp.sqrt(2.0 * wave_spectrum * dw)
    
    # Plot the spectrum
    plt.figure()
    figManager = plt.get_current_fig_manager()
    plt.plot(omega, wave_spectrum)
    plt.xlabel('Angular Frequency [rad/s]')
    plt.ylabel('S(omega)')
    plt.title('JONSWAP Spectrum')
    plt.grid(True)
    plt.show()
    
    # Generate a realization of the sea surface over 100 seconds
    key = jax.random.PRNGKey(0)
    time = jnp.linspace(0, 100, 100*100)
    wave = realization(key, time, jonswap_spectrum, wave_freqs, hs, tp, gamma=gamma, freq_hz=False)
    
    plt.figure()
    figManager = plt.get_current_fig_manager()
    plt.plot(time, wave)
    plt.xlabel('Time [s]')
    plt.ylabel('Wave Elevation [m]')
    plt.title('Wave Elevation Realization')
    plt.grid(True)
    plt.show()
