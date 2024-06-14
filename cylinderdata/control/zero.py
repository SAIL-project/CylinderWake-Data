from typing import Any
from .controller import Controller


class ZeroController(Controller):
    def __init__(self, max_control: float, start_time: float, control_duration: float) -> None:
        super().__init__(max_control, start_time, control_duration)

    def __call__(self, t: float, obs: Any) -> float:
        return 0.0
