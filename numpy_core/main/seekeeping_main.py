import cProfile
import pstats
from numpy_core.gym.mc_gym_csad_numpy import McGym
from numpy_core.controllers.adaptive_seakeeping import MRACShipController
from mclsimpy.utils import Rz, six2threeDOF, three2sixDOF
import numpy as np
import time
def main():
    """Run a simulation using GridWaveEnvironment with a combined MRAC heading + surge PID controller."""

    # (Optional) example of a slowly moving goal over time
    def goal_func(t):
        north0, east0, size0 = 4, 12, 1
        # Make the goal wiggle in north/east over time
        new_north = north0 + 1.0 * np.sin(0.2 * t)
        new_east  = east0  + 0.5 * np.cos(0.1 * t)
        return (new_north, new_east, size0)

    # No moving obstacles in this example
    obstacle_func = None

    # Simulation time step
    dt = 0.08 

    # Create environment
    env = McGym(
        dt=dt,
        grid_width=15,
        grid_height=6,
        render_on=True,    # True => use pygame-based rendering
        final_plot=True    # True => at the end, produce a matplotlib plot of the trajectory
    )

    # Start pose (north=2, east=2, heading=90 deg), facing east
    start_pos = (2, 2, 90)

    # Initial wave conditions (Hs=5, Tp=20, wave_dir=0 deg)
    wave_conditions = (0.05, 1.5, 0)
    
    # Define the goal center and size
    initial_goal = (4, 12, 1)

    # Example static obstacle(s), though set to None below
    # Each obstacle is (obs_n, obs_e, obs_diameter)
    initial_obstacles = [(2, 7, 1.0)]

    # Configure the environment’s task:
    #  - position_tolerance=0.5 => must be within 0.5 m
    #  - goal_heading_deg=90.0  => require final heading = 90 deg
    #  - heading_tolerance_deg=5 => must be within ±5 deg of 90
    env.set_task(
        start_position=start_pos,
        goal=initial_goal,
        wave_conditions=wave_conditions,
        obstacles=initial_obstacles,                # or use initial_obstacles if you want obstacles
        goal_func=None,           # or None if you don’t want a moving goal
        obstacle_func=obstacle_func,   # or None for static obstacles
        position_tolerance=0.5,
        goal_heading_deg=90.0,
        heading_tolerance_deg=40.0
    )

    # Create the MRAC-based controller
    controller = MRACShipController(dt=dt)

    # Total simulation time, steps
    simtime = 150.0
    max_steps = int(simtime / dt)

    print("Starting simulation...")
    start_time = time.time()

    for step_count in range(max_steps):
        # 1) Get the current state from the environment
        state = env.get_state()

        # 2) Compute a control action

        #    The environment’s updated goal is in state["goal"]
        goal_n, goal_e, _ = state["goal"]
        action = controller.compute_action(state, (goal_n, goal_e))

        # 3) Step the environment
        new_state, done, info, reward = env.step(action)

        # 4) (Optional) Check distance to the center of the goal
        boat_n, boat_e, yaw = six2threeDOF(new_state["eta"])
        distance_to_goal = np.sqrt((goal_n - boat_n)**2 + (goal_e - boat_e)**2)
        if distance_to_goal < 0.5:
            print(f"Close to goal at step {step_count}, distance={distance_to_goal:.2f}")

        if done:
            # The environment signaled termination (goal reached w/ heading or collision)
            print("Environment returned done; stopping simulation.")
            break

    total_time = time.time() - start_time
    print(f"Wall-clock time: {total_time:.2f} s")
    print(f"Simulation speed: {(simtime / total_time):.2f}x real-time")
    print("Simulation completed.")

    # After finishing, if final_plot=True, plot the boat trajectory
    env.plot_trajectory()

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(10)  # print top 10 most time-consuming functions