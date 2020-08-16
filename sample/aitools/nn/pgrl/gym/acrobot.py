from .abs_gym import AbsGym
import math
import torch


class Acrobot(AbsGym):
    def __init__(self):
        super().__init__("Acrobot-v1")
        self.maxHeight = -math.inf
        self.onMax = True

    def step(self, action: int):
        obs, r, done = super().step(action)
        r = 0.0

        s = self.env.state
        h = -math.cos(s[0]) - math.cos(s[1] + s[0])
        if h > self.maxHeight:
            self.onMax = True
            self.maxHeight = h
        elif self.onMax:
            self.onMax = False
            r = 1.0

        return obs, torch.tensor(r), done
