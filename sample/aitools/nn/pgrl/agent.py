from .state_action import StateAction
from .i_environment import IEnvironment

import torch


class Agent(object):
    def __init__(self, policyNet):
        self.policy = policyNet

    def __call__(self, env: IEnvironment, render=False):
        obs = env.reset()
        if render:
            env.render()

        done = False

        trajectory = []
        while not done:
            stateAction = StateAction(obs, None, None, None)

            stateAction.prob = self.policy(obs)
            stateAction.action = torch.distributions.Categorical(
                stateAction.prob
            ).sample()

            obs, r, done = env.step(stateAction.action.item())
            stateAction.r = r
            if render:
                env.render()

            trajectory.append(stateAction)

        return trajectory
