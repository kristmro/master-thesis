"""
notebook_demo_jax.py

Demonstration script that initializes vessel parameters and wave conditions,
performs simulations of vessel dynamics using JAX, and visualizes the simulation results. 
Serves as an example and template for vessel simulation workflows.

Author: Kristian Magnus Roen
Date:   2025-03-17
"""

import time
init_start = time.time()
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
#import cProfile, pstats
import numpy as np
# Import functional vessel routines
from jax_core.simulator.vessels.csad_jax import load_csad_parameters, csad_x_dot
# Import the RK4 integrator from your utils
from jax_core.utils import rk4_step
# Import the JONSWAP spectrum from your spectra module
from jax_core.simulator.waves.wave_spectra_jax import jonswap_spectrum
# Import functional wave load routines
from jax_core.simulator.waves.wave_load_jax_jit import init_wave_load, wave_load

#print("JAX devices:", jax.devices())
#profiler = cProfile.Profile()
#profiler.enable()
start = time.time()
# --------------------------------------------------------------------------
# Simulation settings
# --------------------------------------------------------------------------
dt = 0.01
simtime = 100.0
t_array = jnp.arange(0, simtime, dt)
n_steps = t_array.shape[0]

# --------------------------------------------------------------------------
# Load vessel parameters and set up initial state (functional style)
# --------------------------------------------------------------------------
config_file = "/home/kmroen/miniconda3/envs/tensor/lib/python3.9/site-packages/mclsimpy/vessel_data/CSAD/vessel_json.json"
# Load parameters from JSON
params_jit = load_csad_parameters(config_file)


print(f"Init boat took {time.time() - start:.2f} seconds")
# Initial state: x = [eta, nu] (each 6-element)
eta_init = jnp.zeros(6)
nu_init  = jnp.zeros(6)
x = jnp.concatenate([eta_init, nu_init])

# No current
Uc = 0.0
beta_c = 0.0
tau_control = jnp.zeros(6)
start = time.time()
# --------------------------------------------------------------------------
# Define wave parameters & initialize wave load (functional style)
# --------------------------------------------------------------------------
hs = 5.0 / 90.0            # Significant wave height
tp = 10.0 * jnp.sqrt(1/90)    # Peak period
gamma = 3.3                # JONSWAP peak factor

wp = 2 * jnp.pi / tp       # Peak frequency
wmin = 0.5 * wp
wmax = 3.0 * wp

N = 100  # Number of wave components
wave_freqs = jnp.linspace(wmin, wmax, N)  # Frequencies in rad/s



# Compute wave spectrum using our JONSWAP functional spectrum
# (freq_hz=False because wave_freqs are already in rad/s)
omega, wave_spectrum = jonswap_spectrum(wave_freqs, hs, tp, gamma=gamma, freq_hz=False)
dw = (wmax - wmin) / N
wave_amps = jnp.sqrt(2.0 * wave_spectrum * dw)

# Random phases and incident angles
key = jax.random.PRNGKey(0)  # Set the same seed
rand_phase = jax.random.uniform(key, shape=(N,), minval=0, maxval=2*jnp.pi)
#np.random.seed(0)
#rand_phase = np.random.uniform(0, 2*np.pi, size=N)
#print(rand_phase)
#rand_phase = jnp.asarray(rand_phase)
wave_angles = jnp.ones(N) * (jnp.pi / 4)

# Initialize the wave load dictionary (pure functional style)
wl = init_wave_load(
    wave_amps=wave_amps,
    freqs=wave_freqs,
    eps=rand_phase,
    angles=wave_angles,
    config_file=config_file,
    rho=1025,
    g=9.81,
    dof=6,
    depth=100,
    deep_water=True,
    qtf_method="Newman",
    qtf_interp_angles=True,
    interpolate=True
)
print(f"Init the everything related to the waves took {time.time() - start:.2f} seconds")
# --------------------------------------------------------------------------
# Preallocate arrays to store simulation results
# --------------------------------------------------------------------------
eta_result = jnp.zeros((6, n_steps))
nu_result  = jnp.zeros((6, n_steps))

# --------------------------------------------------------------------------
# Simulation loop: use the RK4 integrator and pure functions
# --------------------------------------------------------------------------
print("Starting simulation")
start = time.time()
# Simulation loop with lax.scan
def simulation_step(x, t):
    eta = x[:6]
    tau_wave = wave_load(t, eta, wl)
    tau = tau_control + tau_wave
    x_next = rk4_step(x, dt, csad_x_dot, Uc, beta_c, tau, params_jit)
    return x_next, x_next  # returning the new state as output

x_final, xs = jax.lax.scan(simulation_step, x, t_array)
print(f"Simulation took {time.time() - start:.2f} seconds")
print(f"Simtime/Real: {simtime / (time.time() - start):.2f}")

#profiler.disable()
#stats = pstats.Stats(profiler).sort_stats('cumtime')
#stats.print_stats(20)
# Extract results:
eta_result = xs[:, :6].T   # shape (6, n_steps)
nu_result  = xs[:, 6:].T
print(f"Everything took: {time.time() - init_start:.2f} seconds")
# Plot results:
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
axes[0].plot(t_array, eta_result[0, :], label='x')
axes[0].plot(t_array, eta_result[1, :], label='y')
axes[0].plot(t_array, eta_result[2, :], label='z')
axes[0].legend()
axes[0].grid()
axes[0].set_xlabel('Time [s]')
axes[0].set_ylabel('Position [m]')
axes[0].set_title('Position')

axes[1].plot(t_array, eta_result[5, :], label=r'$\psi$ (yaw)')
axes[1].legend()
axes[1].grid()
axes[1].set_xlabel('Time [s]')
axes[1].set_ylabel('Angle [rad]')
axes[1].set_title('Yaw Angle')

plt.tight_layout()
plt.show()

plt.figure()
plt.plot(eta_result[0, :], eta_result[1, :], label='Trajectory')
plt.grid()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('XY Trajectory')
plt.legend()
plt.show()
