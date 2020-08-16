import torch


class NetworkFF(torch.nn.Module):
    def __init__(self, layerSizes: list, layerActivations: list):
        super(NetworkFF, self).__init__()

        if len(layerSizes) < 2:
            raise Exception("Invalid layer sizes length.")

        if len(layerActivations) != len(layerSizes) - 1:
            raise Exception("Activations/Layer sizes list len missmatch.")

        self.layerActivations = layerActivations

        self.layers = torch.nn.ModuleList()
        for i in range(len(layerSizes) - 1):
            self.layers.append(torch.nn.Linear(layerSizes[i], layerSizes[i + 1]))

    def forward(self, x):
        y = x
        for layer, activation in zip(self.layers, self.layerActivations):
            y = layer(y)
            if activation is not None:
                y = activation(y)

        return y

    def buildOptimizer(self, builder, lr=0.01):
        return builder(params=self.parameters(), lr=lr)
