from typing import Any
from hydrogym.firedrake.utils.pd import PDController as HYGYM_PDController
import numpy as np
from .controller import Controller


class PDController(Controller):
    def __init__(
        self,
        max_control: float,
        start_time: float,
        k: float,
        theta: float,
        dt: float,
        tf: int,
        control_duration: float,
    ) -> None:
        super().__init__(max_control, start_time, control_duration)
        kp = k * np.cos(theta)
        kd = k * np.sin(theta)
        n_steps = int(tf // dt) + 2
        self.controller = HYGYM_PDController(kp, kd, dt, n_steps, filter_type="bilinear", N=20)

    def __call__(self, t: float, obs: Any) -> float:
        if super().__call__(t, obs):
            self.control = self.controller(t, obs)

        return self.control
