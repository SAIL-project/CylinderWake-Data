from typing import Any
import numpy as np
from .controller import Controller


class BaselineController(Controller):
    def __init__(
        self,
        max_control: float,
        start_time: float,
        freq_coeff: float,
        control_duration: float,
    ) -> None:
        super().__init__(max_control, start_time, control_duration)
        self.freq_coeff = freq_coeff

    def __call__(self, t: float, obs: Any) -> float:
        if super().__call__(t, obs):
            # no control period
            if t % 10 > 8:
                self.control = 0.0
            # sinusoidal control
            else:
                freq = 2 * np.pi * (t - self.start_time) / self.freq_coeff
                self.control = self.max_control * np.sin(freq * t)

        return self.control
