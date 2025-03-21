import time
init_start = time.time()
# Imports Gunnerus model
from mclsimpy.simulator import CSAD_DP_6DOF

# Imports waves
from mclsimpy.waves import JONSWAP, WaveLoad

# Imports utilities
import numpy as np

start = time.time()
init_start = time.time()

dt = 0.01
simtime = 100
t = np.arange(0, simtime, dt)

vessel = CSAD_DP_6DOF(dt=dt, method='RK4')
print(f"Init boat took {time.time() - start:.2f} seconds")
Uc = 0.0
beta_c = 0

eta = np.zeros((6, len(t)))
nu = np.zeros((6, len(t)))
tau_control = np.array([0, 0, 0, 0, 0, 0], dtype=float)
start = time.time()
hs = 5.0/90 # Significant wave height
tp = 9.0*np.sqrt(1/90) # Peak period
gamma = 3.3 # Peak factor
wp = 2*np.pi/tp # Peak frequency
wmin = 0.5*wp
wmax = 3.0*wp

N = 100 # Number of wave components

wave_freqs = np.linspace(wmin, wmax, N)

jonswap = JONSWAP(wave_freqs)

_, wave_spectrum = jonswap(hs=hs, tp=tp, gamma=gamma)

dw = (wmax - wmin) / N
wave_amps = np.sqrt(2 * wave_spectrum * dw)
np.random.seed(0)
rand_phase = np.random.uniform(0, 2*np.pi, size=N)
#print(rand_phase)
wave_angles = np.ones(N) * np.pi / 4

waveload = WaveLoad(
    wave_amps=wave_amps,
    freqs=wave_freqs,
    eps=rand_phase,
    angles=wave_angles,
    config_file=vessel._config_file,
    interpolate=True,
    qtf_method="geo-mean",      # Use geometric mean to approximate the QTF matrices.
    deep_water=True,            # Assume deep water conditions.
)

print(f"Init the everything related to the waves took {time.time() - start:.2f} seconds")
eta_init = np.array([0, 0, 0, 0, 0, 0])
nu_init = np.zeros(6)

vessel.set_eta(eta_init)
vessel.set_nu(nu_init)

print("Starting simulation")
start = time.time()
for i in range(len(t)):
    tau_wave = waveload(t[i], vessel.get_eta())
    tau = tau_control + tau_wave
    eta[:, i] = vessel.get_eta()
    nu[:, i] = vessel.get_nu()
    vessel.integrate(Uc, beta_c, tau)


print(f"Simulation took {time.time() - start:.2f} seconds")
print(f"Simtime/Real: {simtime / (time.time() - start):.2f}")
print(f"Everything took: {time.time() - init_start:.2f} seconds")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
axes[0].plot(t, eta[0, :], label='x')
axes[0].plot(t, eta[1, :], label='y')
axes[0].plot(t, eta[2, :], label='z')
axes[0].legend()
axes[0].grid()
axes[0].set_xlabel('Time [s]')
axes[0].set_ylabel('Position [m]')
axes[0].set_title('Position')

axes[1].plot(t, eta[5, :], label=r"$\psi$")
axes[1].legend()
axes[1].grid()
axes[1].set_xlabel('Time [s]')
axes[1].set_ylabel('Angle [rad]')
axes[1].set_title('Angle')

plt.show()


# Plot XY plot to position
plt.figure()
plt.plot(eta[0, :], eta[1, :])
plt.grid()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('XY plot')
plt.show()

