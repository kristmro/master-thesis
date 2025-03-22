#!/usr/bin/env python3
"""
mc_gym_csad_torch.py
TODO: Fix hardcoded cpu device, should be updated to use the device of the input tensors.
TODO: Fix numpy dependecies in the code.

Differentiable MC-Gym for Boat Navigation Simulation.
Replicates the numpy vesion of MC-Gym, but uses PyTorch-based modules:
- CSAD_6DOF (vessel)
- WaveLoad (wave forcing)
- ThrdOrderRefFilter (reference filter)
- utils (utility functions)
- WaveSpectra (for wave spectrum)
    
Author: Kristian Magnus Roen (adapted)
Date:   2025-02-17
"""

import torch
import numpy as np
import pygame
import matplotlib.pyplot as plt

# Adjust these imports to match how/where you've placed the classes.
from torch_core.simulator.vessels.csad_torch import CSAD_6DOF
from torch_core.simulator.waves.wave_load_torch import WaveLoad
from torch_core.simulator.waves.wave_spectra_torch import JONSWAP
from torch_core.ref_gen.reference_filter import ThrdOrderRefFilter
from torch_core.utils import three2sixDOF, six2threeDOF, pipi


class McGym:
    """
    A grid-based environment for boat navigation with wave loads, 
    using differentiable modules for vessel dynamics, wave forcing, etc.

    If 'render_on' is True, uses pygame for real-time display.
    If 'final_plot' is True, logs data for a final matplotlib plot.

    Attributes:
    -----------
    dt : float
        Simulation timestep (seconds).
    grid_width, grid_height : float
        Domain size in meters.
    vessel : CSAD_6DOF
        Differentiable 6-DOF boat simulator.
    waveload : WaveLoad or None
        Differentiable wave load model, or None if no waves.
    curr_sim_time : float
        Current simulation time.
    render_on : bool
        If True, enable pygame visualization.
    final_plot : bool
        If True, log data for a final plot at end.
    trajectory, true_vel : list
        Logs for position and velocity (NumPy).
    start_position : array-like
        (north, east, heading_rad).
    goal : (gN, gE, gSize) or None
        Goal region center + diameter.
    obstacles : list of (oN, oE, oSize)
        Circular obstacles in the domain.
    wave_conditions : (Hs, Tp, waveDirDeg)
        Remembered wave parameters for reference.
    four_corner_test : bool
        If True, uses a preset set of waypoints over time.
    set_points : list
        The corner waypoints if 4-corner test is on.
    simtime : float
        Duration of the 4-corner test in seconds.
    t : array
        Time array for 4-corner test scheduling.
    ref_model : ThrdOrderRefFilter
        Differentiable reference filter for setpoint tracking.
    store_xd : np.ndarray
        Logs reference states at each sim step (for plotting).
    position_tolerance : float
        Acceptable distance to goal to consider "done".
    goal_heading_deg : float or None
        If not None, we also check heading for goal.
    heading_tolerance_deg : float
        Tolerance in degrees for heading match.
    screen, clock : pygame display
        For real-time rendering. 
    """

    def __init__(self,
                 dt=0.1,
                 grid_width=15,
                 grid_height=6,
                 render_on=False,
                 final_plot=True,
                 vessel=CSAD_6DOF):
        # Basic config
        self.dt = dt
        self.grid_width = grid_width
        self.grid_height = grid_height

        # Differentiable vessel simulator
        self.vessel = vessel(dt, method="RK4")
        self.waveload = None
        self.curr_sim_time = 0.0

        # Logging
        self.render_on = render_on
        self.final_plot = final_plot
        self.trajectory = []  # store positions
        self.true_vel = []    # store velocities

        # Task setup
        self.start_position = None
        self.goal = None
        self.obstacles = []
        self.wave_conditions = None
        self.goal_func = None
        self.obstacle_func = None

        # Four-corner test
        self.four_corner_test = False
        self.set_points = None
        self.simtime = 300
        self.t = None
        self.ref_model = None
        self.store_xd = None
        self.ref_omega = None

        # Tolerances
        self.position_tolerance = 0.5
        self.goal_heading_deg = None
        self.heading_tolerance_deg = 10.0

        # Pygame rendering
        self.screen = None
        self.clock = None
        self.WINDOW_WIDTH = 750
        self.WINDOW_HEIGHT = 300
        if self.render_on and pygame is not None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            pygame.display.set_caption("Differentiable Boat Navigation Simulation")
            self.clock = pygame.time.Clock()

        # Transform from domain coords -> screen coords
        self.x_scale = (self.WINDOW_WIDTH  / self.grid_width)
        self.y_scale = (self.WINDOW_HEIGHT / self.grid_height)

        # For RL-like reward or logging
        self.previous_action = np.zeros(3)

    def __del__(self):
        """Ensure resources are closed if environment is deleted."""
        self.close()

    def close(self):
        """Close pygame window if open."""
        if self.render_on and self.screen is not None:
            pygame.quit()

    def get_vessel(self):
        return self.vessel

    def set_task(self,
                 start_position,
                 goal=None,
                 wave_conditions=None,
                 obstacles=None,
                 goal_func=None,
                 obstacle_func=None,
                 position_tolerance=0.5,
                 goal_heading_deg=None,
                 heading_tolerance_deg=10.0,
                 four_corner_test=False,
                 ref_omega=[0.2, 0.2, 0.2],
                 simtime=300):
        """
        Configure a new navigation task or 4-corner test scenario.
        """
        # store start in [n,e,heading(radians)]
        self.start_position = np.array([start_position[0],
                                        start_position[1],
                                        np.deg2rad(start_position[2])])
        self.goal = goal
        self.wave_conditions = wave_conditions
        self.obstacles = obstacles if obstacles is not None else []
        self.goal_func = goal_func
        self.obstacle_func = obstacle_func

        self.position_tolerance = position_tolerance
        self.goal_heading_deg = goal_heading_deg
        self.heading_tolerance_deg = heading_tolerance_deg

        self.four_corner_test = four_corner_test
        if self.four_corner_test:
            # Predefined corner points
            self.set_points = [
                np.array([2.,2.,0.]),
                np.array([4.,2.,0.]),
                np.array([4.,4.,0.]),
                np.array([4.,4.,-np.pi/4]),
                np.array([2.,4.,-np.pi/4]),
                np.array([2.,2.,0.])
            ]
            self.simtime = simtime
            self.t = np.arange(0, self.simtime, self.dt)
            self.ref_omega = ref_omega
            print("Four-corner test enabled.")
            # Differentiable 3rd-order ref filter
            self.ref_model = ThrdOrderRefFilter(self.dt,
                                                    omega=ref_omega,
                                                    initial_eta=torch.tensor(self.start_position,
                                                                             dtype=torch.float32))
            self.store_xd = np.zeros((len(self.t), 9))

        self.reset()

    def reset(self):
        """
        Reset environment and boat state.
        """
        self.curr_sim_time = 0.0

        north0, east0, heading_rad = self.start_position
        # 3->6 DOF
        eta_start = three2sixDOF(torch.tensor([north0, east0, heading_rad],
                                              dtype=torch.float32))
        self.vessel.set_eta(eta_start)

        if self.wave_conditions is not None:
            self.set_wave_conditions(*self.wave_conditions)

        if self.final_plot:
            self.trajectory.clear()
            self.true_vel.clear()

        self.previous_action = np.zeros(3)

        if self.render_on and self.screen is not None:
            self.screen.fill((20,20,20))

    def set_wave_conditions(self, hs, tp, wave_dir_deg, N_w=100, gamma=3.3):
        """
        Build a DiffWaveLoad from wave params: (Hs, Tp, waveDir, etc.)
        """
        self.wave_conditions = (hs, tp, wave_dir_deg)

        # Discretize freq range
        wp = 2.*np.pi / tp
        wmin, wmax = wp/2., 3.*wp
        dw = (wmax - wmin)/N_w
        w = np.linspace(wmin, wmax, N_w)

        # JONSWAP in PyTorch
        jonswap = JONSWAP(w)
        freq, spec = jonswap(hs, tp, gamma)  # freq, spec both Tensors

        wave_amps = np.sqrt(2.0 * spec.detach().cpu().numpy() * dw)
        eps = np.random.uniform(0,2*np.pi, size=N_w)
        wave_dir = np.ones(N_w) * np.deg2rad(wave_dir_deg)

        # DiffWaveLoad expects wave_amps, freqs, eps, angles as NumPy or Torch Tensors
        self.waveload = WaveLoad(
            wave_amps,
            freq.detach().cpu().numpy(),
            eps,
            wave_dir,
            config_file=self.vessel._config_file,
            interpolate=True,
            qtf_method="geo-mean",
            deep_water=True
        )

    def get_four_corner_nd(self, step_count):
        """
        Return reference states at each sim step for the 4-corner test as tensors.
        Additionally, if final_plot is enabled, store a NumPy conversion of eta_d (and _x)
        for plotting purposes.
        """
        current_time = self.t[step_count]

        if self.four_corner_test:
            # Determine setpoint index based on current_time
            if np.allclose(self.start_position, self.set_points[0], atol=1e-3):
                if current_time < 10.0:
                    idx = 0
                else:
                    shifted_time = current_time - 10.0
                    remain_time = self.simtime - 10.0
                    seg_duration = remain_time / 5.0
                    idx = 1 + min(4, int(shifted_time // seg_duration))
            else:
                if current_time > 5 * self.simtime / 6.0:
                    idx = 5
                elif current_time > 4 * self.simtime / 6.0:
                    idx = 4
                elif current_time > 3 * self.simtime / 6.0:
                    idx = 3
                elif current_time > 2 * self.simtime / 6.0:
                    idx = 2
                elif current_time > self.simtime / 6.0:
                    idx = 1
                else:
                    idx = 0

            # Set the desired reference for the filter
            sp = torch.tensor(self.set_points[idx], dtype=torch.float32)
            self.ref_model.set_eta_r(sp)
            self.ref_model.update()

            # Get reference states as tensors (no conversion yet)
            eta_d     = self.ref_model.get_eta_d()
            eta_d_dot = self.ref_model.get_eta_d_dot()
            eta_d_ddot= self.ref_model.get_eta_d_ddot()
            nu_d_body = self.ref_model.get_nu_d()

            # If final_plot is enabled, convert to numpy and store for plotting
            if self.final_plot:
                eta_d_np = eta_d.detach().cpu().numpy()
                
                eta_d_dot_nd = eta_d_dot.detach().cpu().numpy()
                eta_d_ddot_nd= eta_d_ddot.detach().cpu().numpy()
                nu_d_body_nd = nu_d_body.detach().cpu().numpy()
                # For example, store the full state vector for later comparison
                self.store_xd[step_count] = self.ref_model._x.detach().cpu().numpy()
                # Optionally, if you want to store eta_d_np separately, you could:
                # self.eta_d_np_store[step_count] = eta_d_np

            return eta_d, eta_d_dot, eta_d_ddot, nu_d_body
        else:
            # Return zero tensors if not in four-corner test
            zeros_tensor = torch.zeros(3, dtype=torch.float32)
            return zeros_tensor, zeros_tensor, zeros_tensor, zeros_tensor


    def step(self, action):
        """
        Simulate one time step with the given 3DOF action => 6DOF forces + wave.
        Return (state, done, info, reward).
        """
        self.curr_sim_time += self.dt

        # Possibly update dynamic goals or obstacles
        if self.goal_func is not None:
            self.goal = self.goal_func(self.curr_sim_time)
        if self.obstacle_func is not None:
            self.obstacles = self.obstacle_func(self.curr_sim_time)

        # Convert 3DOF -> 6DOF
        tau = torch.tensor(action, dtype=torch.float32)  # ensure matching device
        tau_6 = three2sixDOF(tau)
        tau_6_np = tau_6.detach().cpu().numpy()

        # wave forces
        if self.waveload is not None:
            tau_wave_t = self.waveload(self.curr_sim_time, self.vessel.get_eta())
            tau_wave_np = tau_wave_t.detach().cpu().numpy()
        else:
            tau_wave_np = np.zeros(6)

        total_force_np = tau_6_np + tau_wave_np
        total_force = torch.tensor(total_force_np, dtype=torch.float32)  # Convert to tensor

        # integrate vessel
        self.vessel.integrate(0, 0, total_force)

        # Check termination
        boat_pos = six2threeDOF(self.vessel.get_eta()).detach().cpu().numpy()
        done, info = self._check_termination(boat_pos)

        reward = self.compute_reward(action, self.previous_action)
        self.previous_action = action

        if self.final_plot:
            self.trajectory.append(boat_pos.copy())
            v_3dof = six2threeDOF(self.vessel.get_nu()).detach().cpu().numpy()
            self.true_vel.append(v_3dof.copy())

        if self.render_on:
            self.render()

        return self.get_state(), done, info, reward

    def _check_termination(self, boat_pos):
        """
        Check if we're done:
          - If 4-corner test is done
          - If a goal is reached
          - Or no scenario => done
        """
        if self.four_corner_test:
            if self.curr_sim_time > self.simtime:
                print("Four-corner test completed.")
                return True, {"reason": "four_corner_test_completed"}
            return False, {}

        if self.goal is not None:
            boat_yaw = self.vessel.get_eta()[-1].item()
            g_n, g_e, g_size = self.goal
            dx = boat_pos[0] - g_n
            dy = boat_pos[1] - g_e
            dist_goal = np.sqrt(dx**2 + dy**2)

            heading_ok = True
            if self.goal_heading_deg is not None:
                # normalize
                boat_deg = np.rad2deg(boat_yaw)
                hd = (boat_deg - self.goal_heading_deg +180)%360 -180
                heading_ok = abs(hd) <= self.heading_tolerance_deg

            if dist_goal < self.position_tolerance and heading_ok:
                print("Goal reached!")
                return True, {"reason": "goal_reached"}
            return False, {}
        else:
            # no goal => end
            return True, {"reason": "no_goal"}

    def compute_reward(self, action, prev_action):
        """
        Placeholder reward function for RL or performance-based control.
        """
        return 0.0

    def get_state(self):
        """
        Return dictionary of environment state for external use:
          - 'eta': 6D pose
          - 'nu': 3D velocity
          - 'goal', 'obstacles', 'wave_conditions'
        """
        eta_t = self.vessel.get_eta().detach().cpu().numpy()
        nu3 = six2threeDOF(self.vessel.get_nu()).detach().cpu().numpy()
        return {
            "eta": eta_t,
            "nu": nu3,
            "goal": self.goal,
            "obstacles": self.obstacles,
            "wave_conditions": self.wave_conditions,
        }

    def render(self):
        """
        Pygame-based real-time display of the domain, boat, goal, obstacles.
        """
        if not self.render_on or self.screen is None:
            return
        self.screen.fill((20,20,20))
        self._draw_grid()
        if self.goal is not None:
            self._draw_goal()
        if self.obstacles:
            self._draw_obstacles()
        self._draw_boat()
        pygame.display.flip()
        self.clock.tick(60)

    def _draw_grid(self):
        grid_col = (50,50,50)
        # vertical lines
        for x in range(self.grid_width+1):
            start_px = (x*self.x_scale, 0)
            end_px   = (x*self.x_scale, self.WINDOW_HEIGHT)
            pygame.draw.line(self.screen, grid_col, start_px, end_px, 1)
        # horizontal lines
        for y in range(self.grid_height+1):
            start_px = (0, y*self.y_scale)
            end_px   = (self.WINDOW_WIDTH, y*self.y_scale)
            pygame.draw.line(self.screen, grid_col, start_px, end_px, 1)

    def _draw_goal(self):
        g_n, g_e, g_s = self.goal
        px = g_e*self.x_scale
        py = (self.grid_height - g_n)*self.y_scale
        rad_px = 0.5*g_s*self.x_scale
        pygame.draw.circle(self.screen,(255,215,0),(int(px),int(py)), int(rad_px))

    def _draw_obstacles(self):
        for (obs_n, obs_e, obs_size) in self.obstacles:
            px = obs_e*self.x_scale
            py = (self.grid_height - obs_n)*self.y_scale
            rad_px = 0.5*obs_size*self.x_scale
            pygame.draw.circle(self.screen,(200,0,0),(int(px),int(py)),int(rad_px))

    def _get_boat_hull_local_pts(self):
        """
        Replicate your hull geometry in local coordinates. 
        Then we'll rotate -> global -> screen coords.
        """
        Lpp= 2.5780001
        B  = 0.4440001
        halfL=0.5*Lpp
        halfB=0.5*B
        bow_start_x=0.9344
        def bow_curve_pts(n=40):
            P0=(bow_start_x, +halfB)
            P1=(+halfL, 0.)
            P2=(bow_start_x, -halfB)
            out=[]
            for i in range(n+1):
                s=i/n
                x=(1-s)**2*P0[0]+2*(1-s)*s*P1[0]+s**2*P2[0]
                y=(1-s)**2*P0[1]+2*(1-s)*s*P1[1]+s**2*P2[1]
                out.append((x,y))
            return out
        x_stern_left  = -halfL
        x_stern_right = bow_start_x
        hull=[]
        hull.append((x_stern_left,+halfB))
        hull.append((x_stern_right,+halfB))
        hull.extend(bow_curve_pts(n=40))
        hull.append((x_stern_left,-halfB))
        hull.append((x_stern_left,+halfB))
        # convert snippet => local coords: (snippet_y, snippet_x)
        hull_local= [(p[1],p[0]) for p in hull]
        return np.array(hull_local)

    def _draw_boat(self):
        eta_arr = self.vessel.get_eta().detach().cpu().numpy()
        boat_n, boat_e, boat_yaw = eta_arr[0], eta_arr[1], eta_arr[5]
        hull_local= self._get_boat_hull_local_pts()
        c=np.cos(boat_yaw)
        s=np.sin(boat_yaw)
        rot=np.array([[c,s],[-s,c]])
        pix_pts=[]
        for (lx,ly) in hull_local:
            gx,gy = rot @ np.array([lx,ly])
            gx+=boat_e
            gy+=boat_n
            sx = int(gx*self.x_scale)
            sy = int((self.grid_height-gy)*self.y_scale)
            pix_pts.append((sx,sy))
        pygame.draw.polygon(self.screen,(0,100,255),pix_pts)

    def plot_trajectory(self):
        """
        If final_plot=True, call after done to see trajectory etc.
        """
        if not self.final_plot:
            return
        if len(self.trajectory)==0:
            return

        traj_np = np.array(self.trajectory)
        plt.figure(figsize=(8,4))
        plt.plot(traj_np[:,1], traj_np[:,0],'b-',label="Boat Trajectory")

        if self.goal:
            g_n,g_e,g_s= self.goal
            plt.scatter(g_e, g_n, c='yellow', s=(g_s*self.x_scale)**2,
                        edgecolor='black', label="Goal")

        for obs_n, obs_e, obs_size in self.obstacles:
            plt.scatter(obs_e,obs_n, c='red', s=(obs_size*self.x_scale)**2,
                        edgecolor='black', label="Obstacle")

        plt.xlim([0,self.grid_width])
        plt.ylim([0,self.grid_height])
        plt.xlabel("East [m]")
        plt.ylabel("North [m]")
        plt.title(f"Boat Trajectory in {self.grid_width}×{self.grid_height} domain.")
        plt.legend(loc='upper right', fontsize='small', scatterpoints=1, markerscale=0.1)
        plt.grid(True)
        plt.show()

        if self.four_corner_test:
            # We can also compare self.store_xd vs trajectory
            #  store_xd = [pos(0..2), vel(3..5), acc(6..8)]
            plt.figure(figsize=(8,4))
            plt.plot(traj_np[:,1], traj_np[:,0],'b-', label="Actual Boat Traj")
            plt.plot(self.store_xd[:,1], self.store_xd[:,0],'g-', label="Ref Traj")
            plt.xlabel("East [m]")
            plt.ylabel("North [m]")
            plt.title("Four-Corner Desired vs Actual")
            plt.legend()
            plt.grid(True)
            plt.show()

            # Additional velocity, heading plots as needed.