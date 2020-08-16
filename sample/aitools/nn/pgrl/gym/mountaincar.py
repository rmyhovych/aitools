from .abs_gym import AbsGym
import math
import torch


class MountainCar(AbsGym):
    def __init__(self):
        super().__init__("MountainCar-v0")
        self.maxX = -math.inf
        self.onMax = True

    def step(self, action: int):
        obs, r, done = super().step(action)
        r = 0.0

        x = self.env.state[0]
        if x > self.maxX:
            self.onMax = True
            self.maxX = x
        elif self.onMax:
            self.onMax = False
            r = 1.0

        return obs, torch.tensor(r), done
