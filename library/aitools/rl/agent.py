from .state_action import StateAction
from .i_environment import IEnvironment

import torch


class Agent(object):
    def __init__(self, policy_net):
        self.policy_net = policy_net

    def __call__(self, env: IEnvironment, render=False):
        obs = env.reset()
        if render:
            env.render()

        done = False

        trajectory = []
        while not done:
            state_action = StateAction(obs, None, None, None)

            state_action.prob = self.policy_net(obs)
            state_action.action = torch.distributions.Categorical(
                state_action.prob
            ).sample()

            obs, r, done = env.step(state_action.action.item())
            state_action.r = r
            if render:
                env.render()

            trajectory.append(state_action)

        return trajectory
