import torch
from torch.nn import Linear
from typing import List, Tuple, Callable

from ..network_ff import NetworkFF


class CellLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CellLSTM, self).__init__()
        self.hidden_size = hidden_size

        sub_input_size = input_size + hidden_size
        self.forget_layer = Linear(sub_input_size, hidden_size)

        self.update_layer_force = Linear(sub_input_size, hidden_size)
        self.update_layer_direction = Linear(sub_input_size, hidden_size)

        self.output_layer = Linear(sub_input_size, hidden_size)

    def forward(self, x, h, state):
        xh = torch.cat((x, h))

        new_state = torch.mul(self._forget(xh), state)
        new_state = torch.add(self._update(xh), new_state)
        new_h = torch.mul(self._output(xh), torch.tanh(new_state))

        return new_h, new_state

    def _forget(self, xh):
        return torch.sigmoid(self.forget_layer(xh))

    def _update(self, xh):
        force = torch.sigmoid(self.update_layer_force(xh))
        direction = torch.tanh(self.update_layer_direction(xh))
        return torch.mul(force, direction)

    def _output(self, xh):
        return torch.sigmoid(self.output_layer(xh))


class NetworkLSTM(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_layers: List[Tuple[int, Callable[[torch.Tensor], torch.Tensor]]],
        *,
        device=None
    ):
        super(NetworkLSTM, self).__init__()
        self.device = device

        self.cell = CellLSTM(input_size, hidden_size)

        output_net_builder = NetworkFF.Builder(hidden_size)
        for layer in output_layers:
            output_net_builder.add_layer(layer[0], layer[1])
        self.output_net = output_net_builder.build()

        self.to(self.device)

    def forward(self, xs):
        h, state = self._create_new_states()

        for x in xs:
            h, state = self.cell(x, h, state)

        return self.output_net(h)

    def _create_new_states(self):
        return (
            torch.zeros((self.cell.hidden_size,), device=self.device),
            torch.zeros((self.cell.hidden_size,), device=self.device),
        )
