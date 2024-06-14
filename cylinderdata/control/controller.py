from abc import ABC
from typing import Any


class Controller(ABC):
    def __init__(self, max_control: float, start_time: float, control_duration: float) -> None:
        self.max_control = max_control
        self.start_time = start_time
        self.control_duration = control_duration
        self.last_control = -10
        self.control = 0.0

    def __call__(self, t: float, obs: Any) -> bool:
        if t < self.start_time:
            return False
        elif t - self.last_control > self.control_duration:
            self.last_control = t
            return True
        return False
