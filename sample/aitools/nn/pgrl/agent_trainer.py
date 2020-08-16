from .agent import Agent
from .state_action import StateAction
from .i_environment import IEnvironment

import torch


def PGRL_LOSS(stateAction: StateAction, value=0.0):
    dist = torch.distributions.Categorical(stateAction.prob)
    return -dist.log_prob(stateAction.action) * value


class AgentTrainer(object):
    def __init__(self, agent: Agent, optimizer, loss=PGRL_LOSS, y=0.99):
        self.agent = agent
        self.optimizer = optimizer
        self.loss = loss
        self.y = y

        self.stateActions = []

    def __call__(self, env: IEnvironment):
        trajectory = self.agent(env, render=False)
        self._discountRewards(trajectory, self.y)
        self.stateActions += trajectory
        return trajectory

    def train(self, valueEstimator=None):
        self.optimizer.zero_grad()

        lossAverage = 0.0
        for sa in self.stateActions:
            value = 0.0
            if valueEstimator is not None:
                value = valueEstimator(sa)

            lossPolicy = self.loss(sa, sa.r - value)
            lossAverage += lossPolicy.item()

            lossPolicy.backward(retain_graph=True)
        self.optimizer.step()
        lossAverage /= len(self.stateActions)

        self.stateActions.clear()
        return lossAverage

    def decisiveness(self):
        decisiveness = 0.0
        for prob in [[p.item() for p in list(sa.prob)] for sa in self.stateActions]:
            decisiveness += max(prob)
        decisiveness /= len(self.stateActions)
        return decisiveness

    def _discountRewards(self, trajectory, y):
        cumulativeReturn = 0.0
        for sa in trajectory[::-1]:
            cumulativeReturn = y * cumulativeReturn + sa.r
            sa.r = cumulativeReturn


class VPGTrainer(AgentTrainer):
    def __init__(self, agent, optimizer, y=0.99):
        super().__init__(agent, optimizer, loss=PGRL_LOSS, y=y)
