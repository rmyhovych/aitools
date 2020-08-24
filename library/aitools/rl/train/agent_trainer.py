from .agent import Agent
from .state_action import StateAction
from .i_environment import IEnvironment

import torch


def loss_f(state_action: StateAction, value=0.0):
    dist = torch.distributions.Categorical(state_action.prob)
    return -dist.log_prob(state_action.action) * value


class AgentTrainer(object):
    def __init__(self, agent: Agent, optimizer, y=0.99):
        self.agent = agent
        self.optimizer = optimizer
        self.y = y

        self.state_actions = []

    def __call__(self, env: IEnvironment):
        trajectory = self.agent(env, render=False)
        self._discountRewards(trajectory, self.y)
        self.state_actions += trajectory
        return trajectory

    def train(self, valueEstimator=None):
        self.optimizer.zero_grad()

        lossAverage = 0.0
        for sa in self.state_actions:
            value = 0.0
            if valueEstimator is not None:
                value = valueEstimator(sa)

            loss_policy = loss_f(sa, sa.r - value)
            lossAverage += loss_policy.item()

            loss_policy.backward(retain_graph=True)
        self.optimizer.step()
        lossAverage /= len(self.state_actions)

        self.state_actions.clear()
        return lossAverage

    def decisiveness(self):
        decisiveness = 0.0
        for prob in [[p.item() for p in list(sa.prob)] for sa in self.state_actions]:
            decisiveness += max(prob)
        decisiveness /= len(self.state_actions)
        return decisiveness

    def _discountRewards(self, trajectory, y):
        cumulativeReturn = 0.0
        for sa in trajectory[::-1]:
            cumulativeReturn = y * cumulativeReturn + sa.r
            sa.r = cumulativeReturn


class VPGTrainer(AgentTrainer):
    def __init__(self, agent, optimizer, y=0.99):
        super().__init__(agent, optimizer, y=y)
