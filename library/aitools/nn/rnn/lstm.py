import torch
from typing import List, Tuple, Callable

from ..network_ff import NetworkFF


class CellLSTM(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        memory_size: int,
    ):
        super(CellLSTM, self).__init__()

        sub_input_size = input_size + memory_size

        self.forget_module = self._build_module(
            sub_input_size, memory_size, torch.sigmoid,)

        self.update_module_force = self._build_module(
            sub_input_size, memory_size, torch.sigmoid,)
        self.update_module_direction = self._build_module(
            sub_input_size, memory_size, torch.tanh,)

        self.output_module = self._build_module(
            sub_input_size, memory_size, torch.tanh)

    def forward(self, x, h, state):
        xh = torch.cat((x, h))

        new_state = torch.mul(self._forget(xh), state)
        new_state = torch.add(self._update(xh), new_state)
        new_h = torch.mul(self._output(xh), torch.tanh(new_state))

        return new_h, new_state

    def _forget(self, xh):
        return self.forget_module(xh)

    def _update(self, xh):
        force = self.update_module_force(xh)
        direction = self.update_module_direction(xh)
        return torch.mul(force, direction)

    def _output(self, xh):
        return self.output_module(xh)

    def _build_module(self, in_size: int, out_size: int, out_activation) -> NetworkFF:
        return NetworkFF.Builder(in_size).add_layer(out_size, out_activation)


class NetworkLSTM(torch.nn.Module):
    def __init__(
        self,
        in_size: int,
        memory_size: int,
        out_size: int,
        out_activation,
        *,
        device=None
    ):
        super(NetworkLSTM, self).__init__()
        self.device = device

        self.cell = CellLSTM(in_size, memory_size)
        self.output_net = (
            NetworkFF.Builder(memory_size)
            .add_layer(out_size, out_activation)
            .build()
        )

        self.to(self.device)

    def forward(self, xs):
        h, state = self._create_new_states()

        for x in xs:
            h, state = self.cell(x, h, state)

        return self.output_net(h)

    def _create_new_states(self):
        return (
            torch.zeros((self.cell.memory_size,), device=self.device),
            torch.zeros((self.cell.memory_size,), device=self.device),
        )
