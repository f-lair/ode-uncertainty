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


class CosineAnnealingSchedule(NoiseSchedule):
    """Cosine annealing noise schedule."""

    def __init__(
        self, init_noise_log: float = 0.0, min_noise_log: float = -10.0, cycle_length: int = 4
    ) -> None:
        """
        Initializes noise schedule.

        Args:
            init_noise_log (float, optional): Initial log_10 noise covariance. Defaults to 0.0.
            min_noise_log (float, optional): Minimum log_10 noise covariance. Defaults to -10.0.
            cycle_length (int, optional): Cycle length. Defaults to 4.
        """

        super().__init__(init_noise_log)
        self.min_noise_log = min_noise_log
        self.cycle_length = cycle_length

    def step(self, idx: int) -> Array:
        """
        Performs step in noise schedule.

        Args:
            idx (int): Step index.

        Returns:
            Array: Noise covariance [].
        """

        idx_in_cycle = idx % self.cycle_length
        return jnp.pow(
            10,
            self.min_noise_log
            + 0.5
            * (self.init_noise_log - self.min_noise_log)
            * (1.0 + jnp.cos(idx_in_cycle / (self.cycle_length - 1) * jnp.pi)),
        )
