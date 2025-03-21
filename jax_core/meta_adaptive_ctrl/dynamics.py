
import jax
import jax.numpy as jnp
from jax_core.utils import six2threeDOF, Rz, J, three2sixDOF
from jax_core.simulator.vessels.csad_jax import load_csad_parameters
from jax_core.simulator.waves.wave_spectra_jax import jonswap_spectrum
# Updated import: now use the new jit-compatible module
from jax_core.simulator.waves.wave_load_jax_jit import init_wave_load, WaveLoad

# --------------------------------------------------------------------------
# Load vessel parameters and set up initial state (functional style)
# --------------------------------------------------------------------------

def prior_3dof(q, dq):
    config_file = "/home/kmroen/miniconda3/envs/tensor/lib/python3.9/site-packages/mclsimpy/vessel_data/CSAD/vessel_json.json"
    params_jit = load_csad_parameters(config_file)
    M = six2threeDOF(params_jit["M"])
    D = six2threeDOF(params_jit["D"])
    G = six2threeDOF(params_jit["G"])
    return M, D, G, Rz(q[-1])

def prior_6dof(q, dq):
    config_file = "/home/kmroen/miniconda3/envs/tensor/lib/python3.9/site-packages/mclsimpy/vessel_data/CSAD/vessel_json.json"
    params_jit = load_csad_parameters(config_file)
    M = params_jit["M"]
    D = params_jit["D"]
    G = params_jit["G"]    
    return M, D, G, Rz(q[-1]), J(q)

def plant(q, dq, u, f_ext, prior=prior_3dof):
    M, D, G, R = prior(q, dq)
    if M.shape[0] == 3:
        if f_ext.shape[0] == 3:
            ddq = jax.scipy.linalg.solve(M, f_ext + R @ u - D @ dq - G @ q, assume_a='pos')
        else:
            ddq = jax.scipy.linalg.solve(M, six2threeDOF(f_ext) + R @ u - D @ dq - G @ q, assume_a='pos')
    else:
        if f_ext.shape[0] == 3:
            ddq = jax.scipy.linalg.solve(M, three2sixDOF(f_ext) + three2sixDOF(R @ u) - D @ dq - G @ q, assume_a='pos')
        else:
            ddq = jax.scipy.linalg.solve(M, f_ext + three2sixDOF(R @ u) - D @ dq - G @ q, assume_a='pos')
    return ddq

def disturbance(wave_parm, key, N=15,
                config_file="/home/kmroen/miniconda3/envs/tensor/lib/python3.9/site-packages/mclsimpy/vessel_data/CSAD/vessel_json.json"):
    hs, tp, wave_dir = wave_parm
    print(hs, tp, wave_dir)
    wp = 2 * jnp.pi / tp       # Peak frequency
    wmin = 0.5 * wp
    wmax = 3.0 * wp
    wave_freq = jnp.linspace(wmin, wmax, N)  # Frequencies in rad/s
    omega, wave_spectrum = jonswap_spectrum(wave_freq, hs, tp, gamma=3.3)
    dw = (wmax - wmin) / N
    wave_amps = jnp.sqrt(2.0 * wave_spectrum * dw)
    rand_phase = jax.random.uniform(key, shape=(N,), minval=0, maxval=2 * jnp.pi)
    wave_dir = jnp.deg2rad(wave_dir)
    wave_angles = jnp.ones(N) * wave_dir

    # Initialize wave load using the new, jit-compatible init function.
    wl = init_wave_load(
        wave_amps=wave_amps,
        freqs=omega,
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
    return wl
