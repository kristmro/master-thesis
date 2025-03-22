import torch

class Thruster(torch.nn.Module):
    def __init__(self, pos, K):
        """
        General thruster class.

        Parameters:
        - pos: (tuple or list) (x, y) position of thruster relative to vessel.
        - K: (float) thrust coefficient.
        """
        super().__init__()
        self.register_buffer("_r", torch.tensor(pos, dtype=torch.float32))  # Thruster position (immutable buffer)
        self.register_buffer("_K", torch.tensor([K], dtype=torch.float32))  # Thrust coefficient

    @property
    def pos_x(self):
        """ x-position of thruster. """
        return self._r[0]

    @property
    def pos_y(self):
        """ y-position of thruster. """
        return self._r[1]

    @property
    def K(self):
        """ Thrust coefficient of thruster. """
        return self._K.item()  # Return as Python float for compatibility

class TunnelThruster(Thruster):
    def __init__(self, pos, max_thrust, angle):
        """
        Tunnel Thruster.

        Parameters:
        - pos: (tuple or list) (x, y) position of thruster.
        - max_thrust: (float) Maximum thrust capability.
        - angle: (float) Fixed thrust angle in radians.
        """
        super().__init__(pos, max_thrust)
        self.register_buffer("_angle", torch.tensor([angle], dtype=torch.float32))  # Fixed angle

    @property
    def angle(self):
        """ Get the thruster angle. """
        return self._angle.item()

class AzimuthThruster(Thruster):
    def __init__(self, pos, max_thrust, rotation):
        """
        Azimuth Thruster.

        Parameters:
        - pos: (tuple or list) (x, y) position of thruster.
        - max_thrust: (float) Maximum thrust capability.
        - rotation: (float) Rotational freedom of thruster.
        """
        super().__init__(pos, max_thrust)
        self.register_buffer("_rotation", torch.tensor([rotation], dtype=torch.float32))  # Rotation angle

    @property
    def rotation(self):
        """ Get the thruster rotation angle. """
        return self._rotation.item()