from jax import Array
from jax import numpy as jnp

from src.ode.ode import ODE


class LCAO(ODE):
    """Linearly Coupled Anharmonic Oscillator (second-order)."""

    def __init__(
        self,
        lin_coeff: float = 1.0,
        cubic_coeff: float = 2.0,
        coupling_coeff: float = 0.5,
    ) -> None:
        """
        Initialization for LCAO.

        cf. Steeb, W. H., Louw, J. A., & Villet, C. M. (1987).
        Linearly coupled anharmonic oscillators and integrability.
        Australian journal of physics, 40(5), 587-592.

        Args:
            lin_coeff (float, optional): Coefficient for linear term. Defaults to 1.0.
            cubic_coeff (float, optional): Coefficient for cubic term. Defaults to 2.0.
            coupling_coeff (float, optional): Coefficient for coupling term. Defaults to 0.5.
        """

        self.A = jnp.array(lin_coeff)
        self.a = jnp.array(cubic_coeff)
        self.c = jnp.array(coupling_coeff)

    def fn(self, t: Array, x: Array) -> Array:
        """
        RHS of ODE.
        D=2: Latent dimension.
        N=2: ODE order.
        ...: Batch dimension(s).

        Args:
            t (Array): Time [...].
            x (Array): State [..., N, D].

        Returns:
            Array: d/dt State [..., N, D].
        """

        x_prev = x[..., 0, :]  # [..., D]
        dx_dt_prev = x[..., 1, :]  # [..., D]

        dx_dt_next = dx_dt_prev  # [..., D]
        d2x_dt2_next = (
            -self.A * x_prev - self.a * x_prev**3 - self.c * x_prev[..., [1, 0]]
        )  # [..., D]

        return jnp.stack([dx_dt_next, d2x_dt2_next], axis=-2)  # [..., N, D]
