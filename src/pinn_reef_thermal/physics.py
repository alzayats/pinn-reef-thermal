"""
Reef thermal physics: 1D heat equation, FD solver, source terms.

PDE: ∂T/∂t = κ · ∂²T/∂z² + S(z, t)
  where S(z,t) = (I₀(t)·Kd/ρCp)·exp(-Kd·z) is solar heating

Boundary conditions:
  z=0:     T = T_surface(t)  (Dirichlet — satellite SST)
  z=z_max: ∂T/∂z = 0         (Neumann — insulating bottom)
"""

import numpy as np
from scipy.linalg import solve_banded


class ReefThermalPhysics:
    """Physical parameters and solvers for reef thermal environment."""

    def __init__(self, kappa=5e-4, Kd=0.25, Q_max=350.0, rho_cp=4.1e6,
                 T_mean=28.0, T_amp=1.2, z_max=15.0, t_days=5,
                 utc_offset=10.0):
        self.kappa = kappa
        self.Kd = Kd
        self.Q_max = Q_max
        self.rho_cp = rho_cp
        self.T_mean = T_mean
        self.T_amp = T_amp
        self.z_max = z_max
        self.t_max = t_days * 86400.0
        self.t_days = t_days
        self.utc_offset = utc_offset  # GBR = UTC+10 (AEST)

    def surface_temperature(self, t):
        """Diurnal SST cycle. t in seconds (UTC). Peak at ~14:00 local."""
        local_hour = ((t / 3600.0) + self.utc_offset) % 24.0
        return self.T_mean + self.T_amp * np.sin(2 * np.pi * (local_hour - 8) / 24)

    def solar_source(self, z, t):
        """Depth-dependent solar heating (W/m³ → °C/s). Daytime only."""
        local_hour = ((t / 3600.0) + self.utc_offset) % 24.0
        I_surface = self.Q_max * np.maximum(0, np.sin(np.pi * (local_hour - 6) / 12))
        return (I_surface * self.Kd / self.rho_cp) * np.exp(-self.Kd * z)

    def solve_fd(self, nz=100, dt=60.0):
        """Finite-difference solver (implicit Euler). Returns z, t, T[nz, nt]."""
        dz = self.z_max / (nz - 1)
        nt = int(self.t_max / dt) + 1
        z = np.linspace(0, self.z_max, nz)
        t = np.linspace(0, self.t_max, nt)
        T = np.zeros((nz, nt))
        T[:, 0] = self.T_mean

        r = self.kappa * dt / (dz ** 2)

        for n in range(nt - 1):
            t_next = t[n + 1]
            S = self.solar_source(z, t_next) * dt

            main_diag = np.ones(nz) * (1 + 2 * r)
            upper_diag = np.ones(nz - 1) * (-r)
            lower_diag = np.ones(nz - 1) * (-r)
            rhs = T[:, n].copy() + S

            # Dirichlet BC at surface
            main_diag[0] = 1.0
            upper_diag[0] = 0.0
            rhs[0] = self.surface_temperature(t_next)

            # Neumann BC at bottom
            main_diag[-1] = 1 + r
            lower_diag[-1] = -r
            rhs[-1] = T[-1, n] + S[-1]

            ab = np.zeros((3, nz))
            ab[0, 1:] = upper_diag
            ab[1, :] = main_diag
            ab[2, :-1] = lower_diag
            T[:, n + 1] = solve_banded((1, 1), ab, rhs)

        return z, t, T
