'''
TODO: make the PID controller work with the torch tensors

'''

# pid_main.py
# Not workign just now because the tensor and np confilict in the PID controller
import numpy as np
import time
from torch_core.gym.mc_gym_csad_torch import McGym
from torch_core.controllers.pid import PIDController
from torch_core.utils import six2threeDOF

def main():
    # Environment setup
    dt = 0.08  # Simulation time step
    simtime = 450  # Simulation duration in seconds
    max_steps = int(simtime / dt)

    env = McGym(dt=dt, grid_width=15, grid_height=6, render_on=False, final_plot=True)
    
    # Start position and wave conditions
    start_pos = (2.0, 2.0, 0.0)  # (north, east, heading)
    wave_cond = (5/90, 16/np.sqrt(90), 90.0)  # (Hs=0.5m, Tp=5s, waveDir=90° => from east)

    env.set_task(start_position=start_pos, wave_conditions=wave_cond, four_corner_test=True, simtime=simtime)

    # Initialize simple PID controller
    pid = PIDController(Kp=[10.0, 15.0, 5.0],  # High proportional gain
                        Ki=[0.1, 0.1, 0.05],  # Low integral gain to prevent wind-up
                        Kd=[5.0, 5.0, 2.0],  # Derivative gain for smoothness
                        dt=dt)

    # Run the simulation
    print("Starting four-corner test with PID controller...")
    start_time = time.time()

    for step_count in range(max_steps):
        eta_d, nu_d, eta_d_ddot, nu_d_body = env.get_four_corner_nd(step_count)
        state = env.get_state()

        eta = six2threeDOF(state["eta"])  # Get (north, east, yaw)
        nu = state["nu"]  # Get (u, v, r)

        # Compute control force (tau) using PID
        tau = pid.compute_control(eta, eta_d, nu, nu_d)

        # Apply action and check if done
        _, done, info, _ = env.step(action=tau)

        if done:
            print("Simulation ended:", info)
            break

    total_time = time.time() - start_time
    print(f"Simulation completed in {total_time:.2f} seconds")
    env.plot_trajectory()

if __name__ == "__main__":
    main()