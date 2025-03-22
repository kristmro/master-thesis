import numpy as np

def heading_to_goal(n, e, n_goal, e_goal):
    """
    Compute the desired heading so that 0 rad = facing 'north',
    angles increase toward 'east'. If your environment uses the same
    convention, arctan2(dx, dy) is correct:
       dx = e_goal - e
       dy = n_goal - n
       desired_heading = arctan2(dx, dy).
    """
    dx = (e_goal - e)
    dy = (n_goal - n)
    desired_heading = np.arctan2(dx, dy)
    return desired_heading

class MRACHeadingController:
    """
    A simple MRAC-based heading controller (1 DOF). 
    We use a reference model: 
       Tm * d(psi_m)/dt + psi_m = Km * psi_d
    Then adapt a single gain to drive psi -> psi_m.
    """
    def __init__(self, Tm=5.0, Km=1.0, gamma=0.5, dt=0.1, rudder_gain=1.0):
        """
        Args:
            Tm         : reference model time constant
            Km         : reference model gain
            gamma      : adaptation rate
            dt         : time step
            rudder_gain: factor that translates the 'delta' command into torque
                         if your environment expects yaw torque as action[-1].
        """
        self.Tm = Tm
        self.Km = Km
        self.gamma = gamma
        self.dt = dt
        self.rudder_gain = rudder_gain

        # Reference model heading
        self.psi_m = 0.0
        self.psi_m_dot = 0.0

        # Adaptive parameter
        self.theta_hat = 0.0

    def update(self, psi, psi_d):
        """
        Given current heading psi and desired heading psi_d, compute
        a 'rudder' command delta. We'll then convert that delta to torque.

        1) Tm * psi_m_dot + psi_m = Km * psi_d
        2) e = psi - psi_m
        3) delta = -theta_hat * (psi - psi_d)
        4) theta_hat_dot = gamma * e * (psi - psi_d)
        """
        # --- Reference model update ---
        # Tm * d(psi_m)/dt + psi_m = Km * psi_d
        self.psi_m_dot = (self.Km * psi_d - self.psi_m) / self.Tm
        self.psi_m += self.psi_m_dot * self.dt

        # --- Tracking error ---
        e = psi - self.psi_m

        # --- Regressor: phi = (psi - psi_d)
        phi = (psi - psi_d)

        # --- Adaptation law ---
        theta_hat_dot = self.gamma * e * phi
        self.theta_hat += theta_hat_dot * self.dt

        # --- Control law: delta = -theta_hat * phi
        delta = -self.theta_hat * phi
        return delta

    def get_yaw_torque(self, delta):
        """
        Convert the rudder angle delta to yaw torque.
        If your environment expects torque in action[-1],
        multiply by rudder_gain (tunable).
        """
        return self.rudder_gain * delta


class SurgePID:
    """
    A simple PID (or PD) controller for surge (forward speed).
    For demonstration, we just do P-control: F_surge = Kp * (u_d - u).
    """
    def __init__(self, kp=100.0, dt=0.1, desired_speed=1.0):
        """
        Args:
            kp           : proportional gain
            dt           : time step
            desired_speed: target forward speed in m/s
        """
        self.kp = kp
        self.dt = dt
        self.desired_speed = desired_speed

    def compute_force(self, u):
        """
        Return the surge force needed for (u_d - u).
        """
        error = self.desired_speed - u
        force = self.kp * error
        return force

class MRACShipController:
    """
    Combines heading MRAC + surge PID to produce
    action = [F_surge, 0, M_yaw].
    """
    def __init__(self, dt=0.1):
        self.dt = dt
        # Create the heading MRAC
        self.heading_mrac = MRACHeadingController(Tm=5.0, Km=1.0, gamma=0.5, dt=dt, rudder_gain=50.0)
        # Create surge PID
        self.surge_pid = SurgePID(kp=10.0, dt=dt, desired_speed=1.0)

    def compute_action(self, state, goal_2d):
        """
        Args:
            state   : dict returned by env.get_state():
                      {
                        "boat_position": [n, e],
                        "boat_orientation": yaw_radians,
                        "velocities": [u, v, r],  # forward, lateral, yaw rate
                        ...
                      }
            goal_2d : (goal_n, goal_e)

        Returns:
            action (np.array of shape (3,)): [Fx, Fy, Mz]
        """
        n, e = state["eta"][:2]
        psi = state["eta"][-1]  # heading, radians
        vel = state["nu"]        # [u, v, r]
        u = vel[0]  # forward speed in m/s

        # 1) Compute desired heading
        goal_n, goal_e = goal_2d
        psi_d = heading_to_goal(n, e, goal_n, goal_e)

        # 2) Use MRAC heading controller
        delta = self.heading_mrac.update(psi, psi_d)
        yaw_torque = self.heading_mrac.get_yaw_torque(delta)

        # 3) Use surge PID for forward speed
        surge_force = self.surge_pid.compute_force(u)


        # 4) Construct 3-DOF action = [Fx, Fy, Mz]
        action = np.array([surge_force, 0.0, yaw_torque])
        return action