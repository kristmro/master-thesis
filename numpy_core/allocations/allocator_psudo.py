from abc import ABC, abstractmethod
from itertools import repeat
import numpy as np
import numpy as np
from abc import ABC, abstractmethod
from itertools import repeat
from numpy_core.thruster.thruster import Thruster

DOFS = 3  # Number of degrees of freedom


class AllocationError(Exception):
    """Exception for thrust allocation errors."""

class AllocatorCSAD(ABC):
    """Abstract base class for thruster allocation in CSAD."""

    def __init__(self):
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

    @abstractmethod
    def allocation_problem(self):
        """Constructs the allocation problem matrix."""

    @abstractmethod
    def allocate(self, tau_d):
        """Solves the thrust allocation problem."""

class PseudoInverseAllocator(AllocatorCSAD):

    @property
    def n_problem(self):
        """Number of unknowns to be allocated."""
        return 2 * self.n_thrusters

    def allocation_problem(self):
        """Constructs the allocation matrix and diagonal weighting matrix."""
        if self.n_thrusters == 0:
            raise AllocationError("At least one thruster must be added before allocation.")

        # Construct thrust allocation matrix
        T_e = np.zeros((DOFS, self.n_problem))
        T_e[0, ::2] = 1  # Force in X direction
        T_e[1, 1::2] = 1  # Force in Y direction
        T_e[2, ::2] = [-thruster.pos_y for thruster in self._thrusters]  # Moment
        T_e[2, 1::2] = [thruster.pos_x for thruster in self._thrusters]

        # Construct diagonal gain matrix
        K_vec = np.array([thruster.K for thruster in self._thrusters for _ in range(2)])
        K_e = np.diag(K_vec)

        return T_e, K_e

    def allocate(self, tau_d):
        """Allocates the global thrust vector to the available thrusters."""
        if self.n_thrusters == 0:
            raise AllocationError("At least one thruster must be added before allocation.")

        T_e, K_e = self.allocation_problem()

        # Use pseudo-inverse for numerical stability
        T_e_pseudo = np.linalg.pinv(T_e)

        # Solve for extended thrust vector with element-wise division
        u_e = (1 / np.diag(K_e)) * (T_e_pseudo @ tau_d)

        # Extract magnitude and angle for each thruster
        u = np.hypot(u_e[::2], u_e[1::2])  # sqrt(x^2 + y^2)
        alpha = np.arctan2(u_e[1::2], u_e[::2])  # atan2(y, x)

        return u, alpha