from .abs_gym import AbsGym
import math
import torch


class MountainCar(AbsGym):
    def __init__(self):
        super().__init__("MountainCar-v0")
        self.max_x = -math.inf
        self.on_max = True

    def step(self, action: int):
        obs, r, done = super().step(action)
        r = 0.0

        x = self.env.state[0]
        if x > self.max_x:
            self.on_max = True
            self.max_x = x
        elif self.on_max:
            self.on_max = False
            r = 1.0

        return obs, torch.tensor(r), done
