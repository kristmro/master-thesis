# pid.py
# Not workign just now because the tensor and np confilict in the PID controller
'''TODO: remove the numpy dependency and use torch tensors instead.

'''

import numpy as np
class PIDController:
    """Simple PID controller for 3DOF ship control."""
    def __init__(self, Kp, Ki, Kd, dt):
        """
        Initialize the PID controller.
        
        Parameters:
        - Kp: Proportional gain (array of 3 values for surge, sway, yaw)
        - Ki: Integral gain
        - Kd: Derivative gain
        - dt: Time step
        """
        self.Kp = np.array(Kp)
        self.Ki = np.array(Ki)
        self.Kd = np.array(Kd)
        self.dt = dt

        self.integral_error = np.zeros(3)
        self.prev_error = np.zeros(3)

    def compute_control(self, eta, eta_d, nu, nu_d):
        """
        Compute the PID control force (tau) based on errors.

        Parameters:
        - eta: Current position (north, east, yaw)
        - eta_d: Desired position (north, east, yaw)
        - nu: Current velocity (u, v, r)
        - nu_d: Desired velocity (u, v, r)

        Returns:
        - tau: Control forces (surge, sway, yaw)
        """
        error = eta_d - eta  # Position error
        d_error = nu_d-nu  # Derivative error
        self.integral_error += error * self.dt  # Integral error

        # PID output
        tau = self.Kp * error + self.Ki * self.integral_error + self.Kd * d_error

        # Store previous error for next time step
        self.prev_error = error

        return tau