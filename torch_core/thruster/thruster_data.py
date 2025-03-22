import torch

# ---THRUSTER DATA FOR CSAD (Converted to Torch)---
class ThrusterData:
    """ Container for thruster-related parameters as PyTorch tensors. """
    propeller_diameter = torch.tensor(0.03, dtype=torch.float32)  # [m]
    n_dot_max = torch.tensor(5.0, dtype=torch.float32)            # [1/s^2]
    alpha_dot_max = torch.tensor(2.0, dtype=torch.float32)        # [1/s]
    thrust_max = torch.tensor(1.5, dtype=torch.float32)           # [N]
    thruster_min = torch.tensor(-0.85, dtype=torch.float32)       # [N]

    # Thruster Positions
    lx = torch.tensor([1.0678, 0.9344, 0.9344, -1.1644, -0.9911, -0.9911], dtype=torch.float32)
    ly = torch.tensor([0.0, 0.11, -0.11, 0.0, -0.1644, 0.1644], dtype=torch.float32)

    # Thrust Coefficients
    K = torch.tensor([1.491, 1.491, 1.491, 1.491, 1.491, 1.491], dtype=torch.float32)  # Thrust coefficient

# ----------------------------------------------------------------
# PyTorch-Based Thruster Dynamics for Differentiable Simulation
# ----------------------------------------------------------------