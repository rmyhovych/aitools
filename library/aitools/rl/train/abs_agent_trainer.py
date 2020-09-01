from ..agent_exploring import AgentExploring
from ..i_environment import IEnvironment

import torch


class AbsAgentTrainer(object):
    def __init__(self, agent: AgentExploring, optimizer: torch.optim.Optimizer):
        self.agent = agent
        self.optimizer = optimizer

    def collect_trajectory(self, env: IEnvironment):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def get_decisiveness(self):
        raise NotImplementedError
