from .abs_gym import AbsGym
import math
import torch


class Acrobot(AbsGym):
    def __init__(self):
        super().__init__("Acrobot-v1")
        self.max_height = -math.inf
        self.on_max = True

    def step(self, action: int):
        obs, r, done = super().step(action)
        r = 0.0

        s = self.env.state
        h = -math.cos(s[0]) - math.cos(s[1] + s[0])
        if h > self.max_height:
            self.on_max = True
            self.max_height = h
        elif self.on_max:
            self.on_max = False
            r = 1.0

        return obs, torch.tensor(r), done
