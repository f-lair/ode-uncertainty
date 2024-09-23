from typing import Any, Callable, Dict

from jax import Array
from jax import numpy as jnp

# ODE::(Array:t, Array:x, Dict[str,Array]:params) -> (Array:dx_dt)
ODE = Callable[[Array, Array, Dict[str, Array]], Array]


class ODEBuilder:
    """Abstract builder base class for explicit order-N ODEs."""

    def __init__(self, **kwargs) -> None:
        """
        Initializes ODE builder.
        """

        self.params = {k: jnp.array(v) for k, v in kwargs.items() if isinstance(v, float)}

    def build(self) -> ODE:
        """
        Builds ODE function.

        Raises:
            NotImplementedError: Needs to be defined for a concrete ODE.

        Returns:
            ODE: ODE function.
        """

        raise NotImplementedError

    def build_initial_value(self, initial_value: Array, params: Dict[str, Array]) -> Array:
        """
        Builds initial value in cases in which it is dependent on other factors.
        Defaults to passed initial value.
        D: Latent dimension.
        N: ODE order.

        Args:
            initial_value (Array): Initial value [N, D].
            params (Dict[str, Array]): Parameters.

        Returns:
            Array: Built initial value [N, D].
        """

        return initial_value
