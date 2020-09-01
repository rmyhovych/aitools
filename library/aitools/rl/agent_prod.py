from .i_environment import IEnvironment

import torch


class AgentProd(object):
    def __init__(self, policy_net):
        self.policy_net = policy_net

    def __call__(self, env: IEnvironment):
        obs = env.reset()
        done = False
        while not done:
            prob: torch.Tensor = self.policy_net(obs)
            action = torch.argmax(prob)
            obs, _, done = env.step(action.item())
