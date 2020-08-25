from ..agent import Agent
from ..i_environment import IEnvironment

import torch


class AbsAgentTrainer(object):
    def __init__(self, agent: Agent, optimizer: torch.optim.Optimizer):
        self.agent = agent
        self.optimizer = optimizer

    def collect_trajectory(self, env: IEnvironment):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def get_decisiveness(self):
        raise NotImplementedError
