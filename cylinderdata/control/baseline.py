import numpy as np
from .controller import Controller


class BaselineController(Controller):
    def __init__(self, max_control=0.5):
        print("Baseline Controller initialized")
        self.max_control = max_control

        self.omega = []
        self.time = []

    def control(self, t, obs):
        # Have no inputs for periods, otherwise sinusoidal
        if t % 10 > 9:
            omega = 0.0
        else:
            freq = 2 * np.pi * (t / 10) / 5.0
            omega = self.max_control * np.sin(freq * t)

        # TODO remove
        self.omega.append(omega)
        self.time.append(t)

        return omega
