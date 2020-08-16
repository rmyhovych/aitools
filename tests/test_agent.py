import library.aitools as aitools
import unittest

import random
import torch


class EnvStub(aitools.pgrl.IEnvironment):
    def reset(self):
        return torch.tensor([2.0])

    def step(self, action: int):
        return torch.tensor([2.0]), torch.tensor([1.0]), True

    def render(self):
        pass

    def getSizeObs(self):
        return 1

    def getSizeAction(self):
        return 2


class TestAgent(unittest.TestCase):
    def test_trajectory(self):
        env = EnvStub()

        net = aitools.nn.NetworkFF(
            [env.getSizeObs(), env.getSizeAction()], [torch.nn.functional.softmax]
        )
        agent = aitools.pgrl.Agent(net)

        trajectory = agent(env)

        self.assertEqual(len(trajectory), 1)


if __name__ == "__main__":
    unittest.main()
