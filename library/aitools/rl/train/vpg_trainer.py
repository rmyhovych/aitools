from .abs_agent_trainer import AbsAgentTrainer
from ..agent import Agent
from ..state_action import StateAction
from ..i_environment import IEnvironment

from typing import List

import torch


def vpg_loss_f(probs, actions, values, baselines):
    """
    Vanilla Policy Gradient loss function

    Arguments:
        probs:
            [[0.2, 0.8],
             [0.3, 0.7]]

        actions:
            [0, 1]

        values:
            [0.93, 0.92]

        baselines:
            [0.0, 0.1]
    """
    dist = torch.distributions.Categorical(probs)
    return torch.mean(-dist.log_prob(actions) * (values - baselines))


class VPGTrainer(AbsAgentTrainer):
    def __init__(self, agent: Agent, optimizer: torch.optim.Optimizer, y=0.99):
        super(VPGTrainer, self).__init__(agent=agent, optimizer=optimizer)
        self.y = y
        self.state_actions: List[StateAction] = []

    def collect_trajectory(self, env: IEnvironment):
        trajectory: List[StateAction] = self.agent(env)
        self._discount_rewards(trajectory, self.y)
        self.state_actions += trajectory
        return trajectory

    def train(self, value_baseline_callback):
        probs = torch.stack([sa.prob for sa in self.state_actions])
        actions = torch.cat([sa.action for sa in self.state_actions])
        values = torch.cat([sa.r for sa in self.state_actions])
        baselines = torch.tensor(
            [value_baseline_callback(sa).item() for sa in self.state_actions]
        )

        self.state_actions.clear()

        self.optimizer.zero_grad()
        loss = vpg_loss_f(probs, actions, values, baselines)
        loss.backward()
        self.optimizer.step()

        return loss

    def get_decisiveness(self):
        decisiveness = 0.0
        for prob in [[p.item() for p in list(sa.prob)] for sa in self.state_actions]:
            decisiveness += max(prob)
        decisiveness /= len(self.state_actions)
        return decisiveness

    def _discount_rewards(self, trajectory: List[StateAction], y: float):
        cumulative_return = 0.0
        for sa in trajectory[::-1]:
            cumulative_return = y * cumulative_return + sa.r
            sa.r = cumulative_return
