#------------------------------------------
# Step 1 Import Necessary Libraries
#------------------------------------------
import sys
import numpy as np
import time
from mclsimpy.utils import Rz, six2threeDOF, three2sixDOF
from mclsimpy.simulator.csad import CSAD_DP_6DOF
import numpy_core.allocations.allocator_psudo as al
import numpy_core.thruster.thruster_dynamics as dynamics
import numpy_core.thruster.thruster as thruster
import numpy_core.thruster.thruster_data as data
from numpy_core.gym.mc_gym_csad_numpy import McGym
from numpy_core.controllers.adaptive_fs_controller import AdaptiveFSController

def main():
    # Simulation time step
    dt = 0.08 
    # Total simulation time, steps
    simtime = 450
    max_steps = int(simtime / dt)

    # Start pose 
    start_pos = (2, 2, 0)

    # Initial wave conditions 
    wave_conditions = (0.03 ,1.5 , 45)
    M = CSAD_DP_6DOF(dt)._M 
    print(six2threeDOF(M))
    D = CSAD_DP_6DOF(dt)._D
    N = 512
    # Create environment
    env = McGym(
        dt=dt,
        grid_width=15,
        grid_height=6,
        render_on=True,    # True => use pygame-based rendering
        final_plot=True    # True => at the end, produce a matplotlib plot of the trajectory
    )
    env.set_task(
        start_position=start_pos,
        wave_conditions=wave_conditions,
        four_corner_test=True,
        simtime=simtime,
        ref_omega=[0.2, 0.2, 0.02]
    )
    # Create the PD-based controller
    controller=AdaptiveFSController(dt=dt, M=M, D=D, N=N)
    allocator = al.PseudoInverseAllocator()

    for i in range(6):
        allocator.add_thruster(thruster.Thruster(pos=[data.lx[i], data.ly[i]], K=data.K[i]))

    # Start the simulation
    print("Starting simulation...")
    start_time = time.time()
    u_stored = [np.zeros(6)]  
    ThrustDyn = dynamics.ThrusterDynamics()
    for step_count in range(max_steps):
        eta_d, nu_d, eta_d_ddot, nu_d_body = env.get_four_corner_nd(step_count)
        state = env.get_state()
        #nu_d = Rz(state["eta"][-1]) @ nu_d
        tau, debug = controller.get_tau(eta=six2threeDOF(state["eta"]),eta_d=eta_d, nu= state["nu"], eta_d_dot=nu_d, eta_d_ddot= eta_d_ddot, t=step_count*dt, calculate_bias=True)
        u, alpha = allocator.allocate(tau)
        u_stored.append(u)
        u = ThrustDyn.limit_rate(u, u_stored[-2], data.alpha_dot_max, dt)
        u = ThrustDyn.saturate(u, data.thruster_min, data.thrust_max)  

        tau_cmd = ThrustDyn.get_tau(u, alpha)
        # Print some information
        if step_count % 100 == 0:  # Since dt=0.1, this prints every 10 seconds
            print("\nControl forces and commands:")
            print(f"tau    = [{tau[0]:8.2f}, {tau[1]:8.2f}, {tau[2]:8.2f}]")
            print(f"tau_cmd= [{tau_cmd[0]:8.2f}, {tau_cmd[1]:8.2f}, {tau_cmd[2]:8.2f}]")
            print("\nThruster settings:")
            print(f"Thrust magnitudes (u) = {[f'{x:6.2f}' for x in u]}")
            print(f"Thrust angles (alpha) = {[f'{x:6.2f}' for x in alpha]}")
            print("\nDebug information:")
            print(f'b_hat: {debug["b_hat"]}')
            print(f'tau_z2: {debug["tau_z2"]}')
            print(f'tau_alpha: {debug["tau_alpha"]}')
            print(f'tau_alpha_dot: {debug[ "tau_alpha_dot"]}')
            print(f'z1: {debug["z1"]}')
            print(f'z2: {debug["z2"]}')
        _, done, info, _ = env.step(action = tau_cmd)
        if done:
            # The environment signaled termination (goal reached w/ heading or collision)
            print("Environment returned done; stopping simulation, because", info)
            break
    total_time = time.time() - start_time
    print(f"Wall-clock time: {total_time:.2f} s")
    print(f"Simulation speed: {(simtime / total_time):.2f}x real-time")
    print("Simulation completed.")
    # After finishing, if final_plot=True, plot the boat trajectory
    env.plot_trajectory()
        


if __name__ == "__main__":
    main()