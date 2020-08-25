import torch

from .abs_baseline_provider import AbsBaselineProvider


class ZeroBaselineProvider(AbsBaselineProvider):
    def __call__(self, x):
        return torch.zeros((1,))

    def update(self):
        pass
