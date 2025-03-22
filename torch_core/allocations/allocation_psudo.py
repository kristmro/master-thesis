import torch

DOFS = 3  # Number of degrees of freedom

class AllocationError(Exception):
    """Exception for thrust allocation errors."""

class AllocatorCSAD(torch.nn.Module):
    """Abstract base class for thruster allocation in CSAD."""
    
    def __init__(self):
        super().__init__()
        self._thrusters = []

    @property
    def n_thrusters(self):
        """Returns the number of thrusters."""
        return len(self._thrusters)

    def add_thruster(self, thruster):
        """Adds a thruster to the allocation problem."""
        if hasattr(thruster, "pos_x") and hasattr(thruster, "pos_y") and hasattr(thruster, "K"):
            self._thrusters.append(thruster)
        else:
            raise TypeError("Thruster must have pos_x, pos_y, and K attributes.")

    def allocation_problem(self):
        """Constructs the allocation problem matrix."""
        raise NotImplementedError

    def allocate(self, tau_d):
        """Solves the thrust allocation problem."""
        raise NotImplementedError

class PseudoInverseAllocator(AllocatorCSAD):
    """Pseudo-inverse based thrust allocation for CSAD."""

    @property
    def n_problem(self):
        """Number of unknowns to be allocated."""
        return 2 * self.n_thrusters  # Each thruster has (Fx, Fy)

    def allocation_problem(self):
        """Constructs the allocation matrix and diagonal weighting matrix."""
        if self.n_thrusters == 0:
            raise AllocationError("At least one thruster must be added before allocation.")

        # Construct thrust allocation matrix (T_e) as a torch tensor
        T_e = torch.zeros((DOFS, self.n_problem), dtype=torch.float32)

        # Force in X and Y directions
        T_e[0, ::2] = 1  # Surge force (x-component)
        T_e[1, 1::2] = 1  # Sway force (y-component)

        # Moment (torque) due to thruster placement
        thruster_positions_x = torch.tensor([thruster.pos_x for thruster in self._thrusters], dtype=torch.float32)
        thruster_positions_y = torch.tensor([thruster.pos_y for thruster in self._thrusters], dtype=torch.float32)
        T_e[2, ::2] = -thruster_positions_y  # Contribution from x-forces
        T_e[2, 1::2] = thruster_positions_x  # Contribution from y-forces

        # Construct diagonal gain matrix (K_e)
        K_vec = torch.tensor([thruster.K for thruster in self._thrusters for _ in range(2)], dtype=torch.float32)
        K_e = torch.diag(K_vec)  # Create diagonal matrix

        return T_e, K_e

    def allocate(self, tau_d):
        """
        Allocates the global thrust vector to the available thrusters using the pseudo-inverse method.
        
        Parameters:
        - tau_d (torch.Tensor): Desired force/moment vector (shape: [3])
        
        Returns:
        - u (torch.Tensor): Thruster magnitudes (shape: [n_thrusters])
        - alpha (torch.Tensor): Thruster angles (shape: [n_thrusters])
        """
        if self.n_thrusters == 0:
            raise AllocationError("At least one thruster must be added before allocation.")

        T_e, K_e = self.allocation_problem()

        # Use Moore-Penrose pseudo-inverse for numerical stability
        T_e_pseudo = torch.linalg.pinv(T_e)  # Computes the pseudo-inverse of T_e

        # Compute extended thrust vector
        u_e = torch.linalg.solve(K_e, T_e_pseudo @ tau_d)  # Solve Ku = T_e_pseudo @ tau_d

        # Compute magnitude (u) and angle (alpha) of thrust vectors
        u = torch.hypot(u_e[::2], u_e[1::2])  # Magnitude: sqrt(Fx^2 + Fy^2)
        alpha = torch.atan2(u_e[1::2], u_e[::2])  # Angle: atan2(Fy, Fx)

        return u, alpha