from jax import Array
from jax import numpy as jnp


class NoiseSchedule:
    """Abstract base class for noise schedule."""

    def __init__(self, init_noise_log: float) -> None:
        """
        Initializes noise schedule.

        Args:
            init_noise_log (float): Initial log_10 noise covariance.
        """

        self.init_noise_log = init_noise_log

    def step(self, idx: int) -> Array:
        """
        Performs step in noise schedule.

        Args:
            idx (int): Step index.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete noise schedule.

        Returns:
            Array: Noise covariance [].
        """

        raise NotImplementedError


class LinearDecaySchedule(NoiseSchedule):
    """Linear noise decay."""

    def __init__(self, init_noise_log: float = 0.0, decay_rate: float = 1.0) -> None:
        """
        Initializes noise schedule.

        Args:
            init_noise_log (float, optional): Initial log_10 noise covariance. Defaults to 0.0.
            decay_rate (float, optional): Decay rate. Defaults to 1.0.
        """

        super().__init__(init_noise_log)
        self.decay_rate = decay_rate

    def step(self, idx: int) -> Array:
        """
        Performs step in noise schedule.

        Args:
            idx (int): Step index.

        Returns:
            Array: Noise covariance [].
        """

        return jnp.pow(10, self.init_noise_log - idx * self.decay_rate)


class ExponentialDecaySchedule(NoiseSchedule):
    """Exponential noise decay."""

    def __init__(self, init_noise_log: float = 0.0, decay_rate: float = 8.0) -> None:
        """
        Initializes noise schedule.

        Args:
            init_noise_log (float, optional): Initial log_10 noise covariance. Defaults to 0.0.
            decay_rate (float, optional): Decay rate. Defaults to 1.0.
        """

        super().__init__(init_noise_log)
        self.decay_rate = decay_rate

    def step(self, idx: int) -> Array:
        """
        Performs step in noise schedule.

        Args:
            idx (int): Step index.

        Returns:
            Array: Noise covariance [].
        """

        return jnp.pow(10, self.init_noise_log - self.decay_rate * jnp.log10(idx + 1))
