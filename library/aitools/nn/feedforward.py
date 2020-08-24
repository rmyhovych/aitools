import torch
from typing import List


class Linear(torch.nn.Module):
    def __init__(self, in_size, out_size, activation=None):
        super(Linear, self).__init__()

        self.layer = torch.nn.Linear(in_size, out_size)
        self.activation = activation
        if self.activation is None:
            self.activation = self._no_activation

    def forward(self, x):
        return self.activation(self.layer(x))

    def _no_activation(tensor):
        return tensor


class NetworkFF(torch.nn.Module):
    class Builder(object):
        def __init__(self, in_size):
            self.prev_out_size = in_size
            self.layers: List[Linear] = []

        def add_layer(self, out_size, activation=None):
            self.layers.append(
                Linear(self.prev_out_size, out_size, activation=activation)
            )
            self.prev_out_size = out_size
            return self

        def build(self):
            return NetworkFF(self.layers)

    def __init__(self, layers: List[Linear]):
        super(NetworkFF, self).__init__()

        if len(layers) < 1:
            raise Exception("Invalid number of layers.")

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y

    def build_optimizer(self, builder, lr=0.01):
        return builder(params=self.parameters(), lr=lr)


if __name__ == "__main__":
    net = (
        NetworkFF.Builder(2)
        .add_layer(10000, torch.nn.functional.relu)
        .add_layer(2, torch.sigmoid)
        .build()
    )
    opt = net.build_optimizer(torch.optim.Adam, lr=0.01)

    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    label = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    for _ in range(20):
        y = net(x)
        loss = torch.nn.functional.mse_loss(y, label)
        print(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()
