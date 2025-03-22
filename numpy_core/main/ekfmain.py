import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from mclsimpy.simulator.csad import CSAD_DP_6DOF
from numpy_core.observers.ekf import EKF
from mclsimpy.waves.wave_loads import WaveLoad
from mclsimpy.waves.wave_spectra import JONSWAP

from mclsimpy.utils import six2threeDOF, three2sixDOF

# Simulation parameters
dt = 0.01  # Time step
N = 10000  # Number of steps
t = np.arange(0, dt * N, dt)
np.random.seed(1234)  # Seed for reproducibility

# Sea state parameters
N_w = 25  # Number of wave components
hs = 2.5  # Significant wave height
tp = 9.0  # Peak period
wp = 2 * np.pi / tp  # Peak frequency
wmin = wp / 2
wmax = 2.5 * wp
dw = (wmax - wmin) / N_w
w = np.linspace(wmin, wmax, N_w)

# Create wave spectrum
jonswap = JONSWAP(w)
freq, spec = jonswap(hs, tp, gamma=1.8)

# Wave amplitudes and directions
wave_amps = np.sqrt(2 * spec * dw)
eps = np.random.uniform(0, 2 * np.pi, size=N_w)  # Random phase
wave_dir = np.ones(N_w) * np.deg2rad(180)  # Wave direction: 180 degrees

# Create simulation objects
vessel = CSAD_DP_6DOF(dt)  # Vessel dynamics
KalmanFilter = EKF(dt, vessel._M, vessel._D, Tp=tp)  # Kalman filter
waveload = WaveLoad(wave_amps, w, eps, wave_dir, config_file=vessel._config_file)  # Wave loads

# Storage for results
eta_storage = np.zeros((N, 3))  # True state in NED
eta_hat_storage = np.zeros((N, 3))  # Kalman filter estimates

# Simulation loop
for i in tqdm(range(N)):
    # Time
    current_time = (i + 1) * dt

    # Wave forces
    tau_wf = waveload.first_order_loads(current_time, vessel.get_eta())
    tau_sv = waveload.second_order_loads(current_time, vessel.get_eta()[-1])
    tau_w = tau_wf + tau_sv  # Total wave forces

    # Apply forces and integrate dynamics
    vessel.integrate(0, 0, tau_w)  # No control forces applied

    # True state
    eta_storage[i] = vessel.get_eta()[:3]  # True NED position

    # Measurements with noise
    noise = np.concatenate((np.random.normal(0, .167, size=3), np.random.normal(0, .017, size=3)))
    y = np.array(vessel.get_eta() + noise)  # Noisy measurements for Kalman filter

    # Kalman filter update
    KalmanFilter.update(np.zeros(3), six2threeDOF(y))  # No control forces used, tau_cmd=0

    # Kalman filter estimate
    eta_hat_storage[i] = KalmanFilter.get_eta_hat()  # Kalman filter estimated NED position

# Visualization: True vs. Kalman Filter NED Positions
plt.figure(figsize=(10, 12))

# North-East Trajectory
plt.subplot(3, 1, 1)
plt.plot(eta_storage[:, 1], eta_storage[:, 0], label="True Trajectory (East vs North)", color='blue')
plt.plot(eta_hat_storage[:, 1], eta_hat_storage[:, 0], '--', label="Kalman Trajectory (East vs North)", color='orange')
plt.xlabel("East (m)")
plt.ylabel("North (m)")
plt.legend()
plt.grid()
plt.title("Vessel Trajectory in NED Coordinates")

# North Position Over Time
plt.subplot(3, 1, 2)
plt.plot(t, eta_storage[:, 0], label="True North", color='blue')
plt.plot(t, eta_hat_storage[:, 0], '--', label="Kalman North", color='orange')
plt.xlabel("Time (s)")
plt.ylabel("North Position (m)")
plt.legend()
plt.grid()
plt.title("North Position Over Time")

# East Position Over Time
plt.subplot(3, 1, 3)
plt.plot(t, eta_storage[:, 1], label="True East", color='blue')
plt.plot(t, eta_hat_storage[:, 1], '--', label="Kalman East", color='orange')
plt.xlabel("Time (s)")
plt.ylabel("East Position (m)")
plt.legend()
plt.grid()
plt.title("East Position Over Time")

plt.tight_layout()
plt.show()