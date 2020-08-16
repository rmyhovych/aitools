from .. import IEnvironment

import gym
import torch


class AbsGym(IEnvironment):
    def __init__(self, gymtype):
        self.env = gym.make(gymtype)

    def __del__(self):
        self.env.close()

    def reset(self):
        return self._obsToTensor(self.env.reset())

    def step(self, action: int):
        obs, r, done, _ = self.env.step(action)
        return self._obsToTensor(obs), torch.tensor([r]), done

    def render(self):
        self.env.render()

    def getSizeObs(self):
        return self.env.observation_space.shape[0]

    def getSizeAction(self):
        return self.env.action_space.n

    def _obsToTensor(self, obs):
        return torch.from_numpy(obs).type(dtype=torch.float)
