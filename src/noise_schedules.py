from jax import Array
from jax import numpy as jnp


class NoiseSchedule:
    """Abstract base class for observation noise schedule."""

    def __init__(self, init_noise_log: float) -> None:
        self.init_noise_log = init_noise_log

    def step(self, idx: int) -> Array:
        raise NotImplementedError


class LinearDecaySchedule(NoiseSchedule):
    def __init__(self, init_noise_log: float = 0.0, decay_rate: float = 1.0) -> None:
        super().__init__(init_noise_log)
        self.decay_rate = decay_rate

    def step(self, idx: int) -> Array:
        return jnp.pow(10, self.init_noise_log - idx * self.decay_rate)


class ExponentialDecaySchedule(NoiseSchedule):
    def __init__(self, init_noise_log: float = 0.0, decay_rate: float = 8.0) -> None:
        super().__init__(init_noise_log)
        self.decay_rate = decay_rate

    def step(self, idx: int) -> Array:
        return jnp.pow(10, self.init_noise_log - self.decay_rate * jnp.log10(idx + 1))
