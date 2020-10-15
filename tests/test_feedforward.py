import aitools
import unittest

import random
import torch


class TestFeedforward(unittest.TestCase):
    def test_input_size(self):
        IN_SIZE = random.randint(1, 10)

        net = aitools.nn.NetworkFF([IN_SIZE, 10], [None])
        firstLayer = net.layers[0]

        actualInput = firstLayer.in_features
        self.assertEqual(actualInput, IN_SIZE)

        inData = torch.randn((IN_SIZE,))
        net(inData)

    def test_input_size_negative(self):
        IN_SIZE = random.randint(1, 10)

        net = aitools.nn.NetworkFF([IN_SIZE, 10], [None])

        IN_DATA_SIZE = random.choice([i for i in range(1, 11) if i != IN_SIZE])
        inData = torch.randn((IN_DATA_SIZE,))
        self.assertRaises(RuntimeError, lambda: net(inData))

    def test_output_size(self):
        OUT_SIZE = random.randint(1, 10)

        inData = torch.randn((10,))
        net = aitools.nn.NetworkFF([inData.size()[0], OUT_SIZE], [None])
        outData = net(inData)

        self.assertEqual(outData.size(), (OUT_SIZE,))


if __name__ == "__main__":
    unittest.main()
