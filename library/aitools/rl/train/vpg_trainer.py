from .abs_agent_trainer import AbsAgentTrainer
from ..agent import Agent
from ..state_action import StateAction
from ..i_environment import IEnvironment

from typing import List

import torch


def vpg_loss_f(state_action: StateAction, value=0.0):
    dist = torch.distributions.Categorical(state_action.prob)
    return -dist.log_prob(state_action.action) * value


class VPGTrainer(AbsAgentTrainer):
    def __init__(self, agent: Agent, optimizer: torch.optim.Optimizer, y=0.99):
        super(AbsAgentTrainer, self).__init__(agent=agent, optimizer=optimizer)
        self.y = y
        self.state_actions: List[StateAction] = []

    def collect_trajectory(self, env: IEnvironment):
        trajectory: List[StateAction] = self.agent(env)
        self._discount_rewards(trajectory, self.y)
        self.state_actions += trajectory
        return trajectory

    def train(self, value_baseline_callback):
        self.optimizer.zero_grad()

        loss_average = 0.0
        for sa in self.state_actions:
            baseline = value_baseline_callback(sa)
            loss_policy = vpg_loss_f(sa, sa.r - baseline)

            loss_average += loss_policy.item()

            loss_policy.backward(retain_graph=True)
        self.optimizer.step()
        loss_average /= len(self.state_actions)

        self.state_actions.clear()
        return loss_average

    def get_decisiveness(self):
        decisiveness = 0.0
        for prob in [[p.item() for p in list(sa.prob)] for sa in self.state_actions]:
            decisiveness += max(prob)
        decisiveness /= len(self.state_actions)
        return decisiveness

    def _discount_rewards(self, trajectory: List[StateAction], y: float):
        cumulative_return = 0.0
        for sa in trajectory[::-1]:
            cumulativeReturn = y * cumulative_return + sa.r
            sa.r = cumulativeReturn
