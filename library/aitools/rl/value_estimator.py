from .state_action import StateAction

import torch.nn.functional as F


class ValueEstimator(object):
    def __init__(self, net, optimizer, loss=F.mse_loss):
        self.net = net
        self.optimizer = optimizer
        self.loss = loss
        self.optimizer.zero_grad()

    def __call__(self, stateAction: StateAction):
        value = self.net(stateAction.obs)
        lossValue = self.loss(value, stateAction.r)
        lossValue.backward(retain_graph=True)

        return value

    def train(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
