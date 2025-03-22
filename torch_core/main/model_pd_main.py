#!/usr/bin/env python3
"""
Main script to run a scaled simulation with model-based control.

Scaling applied:
- λ = 1/90
- Time scaled by sqrt(λ)
- Wave height scaled by λ

Model-based control:
- Uses simple PD feedback control.
- Integrates a reference tracking strategy.

Author: Kristian Magnus Roen
Date: 19.02.2025
"""

import torch
import numpy as np
import time
from torch_core.gym.mc_gym_csad_torch import McGym
from torch_core.controllers.model_based import ModelController
from torch_core.allocations.allocation_psudo import PseudoInverseAllocator
from torch_core.thruster.thruster_data import ThrusterData
from torch_core.thruster.thruster_dynamics import ThrusterDynamics
from torch_core.thruster.thruster import Thruster


# -----------------------------
#  Constants and Scaling Factors
# -----------------------------
LAMBDA = 1 / 90  # Model scaling factor
TIME_SCALE = np.sqrt(LAMBDA)  # Time scaling
WAVE_HEIGHT_SCALE = LAMBDA  # Scale wave height accordingly

def main():
    # -----------------------------
    #  Simulation Parameters
    # -----------------------------
    dt = 0.1   # Time step 
    simtime = 450   # Total simulation
    start_pos = (2.0, 2.0, 0.0)  # (north, east, heading)

    # Define wave conditions
    wave_conditions = (2.0 * WAVE_HEIGHT_SCALE, 8.0 * TIME_SCALE, 180.0)  # Scaled wave properties

    # -----------------------------
    #  Initialize Environment
    # -----------------------------
    env = McGym(dt=dt,grid_width=15,
                                  grid_height=6,
                                  render_on=True,
                                  final_plot=True
                                  )
    env.set_task(start_pos, 
                 goal=None, 
                 wave_conditions=wave_conditions, 
                 four_corner_test=True, 
                 simtime=simtime,
                 ref_omega=[0.3, 0.3, 0.15]
                 )

    # -----------------------------
    #  Initialize Model-Based Controller
    # -----------------------------
    controller = ModelController(dt=dt)

    # -----------------------------
    #  Initialize Thrust Allocation
    # -----------------------------
    allocator = PseudoInverseAllocator()
    for i in range(6):
        allocator.add_thruster(Thruster(pos=[ThrusterData.lx[i], ThrusterData.ly[i]], K=ThrusterData.K[i]))

    thrust_dynamics = ThrusterDynamics()
    u_stored = [torch.zeros(6)]

    # -----------------------------
    #  Simulation Loop
    # -----------------------------
   
    for step_count in range(int(simtime/dt)):

        # Get reference trajectory from four-corner test
        eta_d, nu_d, eta_d_ddot, _ = env.get_four_corner_nd(step_count)
        state = env.get_state()
        
        # Compute control forces using the model-based controller
        tau = controller.compute_control(state, eta_d, nu_d, eta_d_ddot)

        # Allocate thrust using pseudo-inverse method
        u, alpha = allocator.allocate(tau)
        u_stored.append(u)
        u = thrust_dynamics.limit_rate(u, u_stored[-2], ThrusterData.alpha_dot_max, dt)
        u = thrust_dynamics.saturate(u, ThrusterData.thruster_min, ThrusterData.thrust_max)

        
        # Compute actual forces applied to the boat
        tau_cmd = thrust_dynamics.get_tau(u, alpha)

        if step_count % 100 == 0:  # Since dt=0.1, this prints every 10 seconds
            print("\nControl forces and commands:")
            print(f"tau    = [{tau[0]:8.2f}, {tau[1]:8.2f}, {tau[2]:8.2f}]")
            print(f"tau_cmd= [{tau_cmd[0]:8.2f}, {tau_cmd[1]:8.2f}, {tau_cmd[2]:8.2f}]")
            print("\nThruster settings:")
            print(f"Thrust magnitudes (u) = {[f'{x:6.2f}' for x in u]}")
            print(f"Thrust angles (alpha) = {[f'{x:6.2f}' for x in alpha]}")

        # Step simulation
        _, done, info, _ = env.step(action=tau)
       

    # -----------------------------
    #  Post-Simulation Analysis
    # -----------------------------
    env.plot_trajectory()
    print("Simulation complete.")


if __name__ == "__main__":
    main()