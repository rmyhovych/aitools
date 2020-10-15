import torch

from .abs_baseline_provider import AbsBaselineProvider
from ..state_action import StateAction


class ValueEstimator(AbsBaselineProvider):
    def __init__(self, net, optimizer, loss_f=torch.nn.functional.mse_loss):
        self.net = net
        self.optimizer = optimizer
        self.loss_f = loss_f

        self.estimated_values = []
        self.actual_values = []

    def __call__(self, state_action: StateAction):
        estimated_value = self.net(state_action.obs)

        self.estimated_values.append(estimated_value)
        self.actual_values.append(state_action.r)

        return estimated_value

    def update(self) -> float:
        ys = torch.stack(self.estimated_values)
        self.estimated_values.clear()

        labels = torch.stack(self.actual_values)
        self.actual_values.clear()

        self.optimizer.zero_grad()
        loss = self.loss_f(ys, labels)
        loss.backward()
        self.optimizer.step()

        return loss.item()
