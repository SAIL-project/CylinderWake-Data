from abc import ABC


class Controller(ABC):
    def __init__(self):
        print("Dummy Controller initialized")

    def control(self, t, obs) -> float:
        return 0.0
