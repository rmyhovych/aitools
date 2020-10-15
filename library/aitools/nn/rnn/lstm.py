import torch
from typing import List, Tuple, Callable

from ..network_ff import NetworkFF


class CellLSTM(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        memory_size: int,
        hidden_sizes: List[Tuple[int, Callable[[torch.Tensor], torch.Tensor]]],
    ):
        super(CellLSTM, self).__init__()
        self.memory_size = memory_size

        sub_input_size = input_size + memory_size

        self.forget_module = self._build_module(sub_input_size, hidden_sizes + [memory_size, torch.sigmoid])

        self.update_module_force = self._build_module(sub_input_size, hidden_sizes + [memory_size, torch.sigmoid])
        self.update_module_direction = self._build_module(sub_input_size, hidden_sizes + [memory_size, torch.tanh])

        self.output_module = self._build_module(sub_input_size, hidden_sizes + [memory_size, torch.sigmoid])

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

    def _build_module(self, in_size: int, layers: List[Tuple[int, Callable[[torch.Tensor], torch.Tensor]]]) -> NetworkFF:
        builder = NetworkFF.Builder(in_size)
        for layer in layers:
            builder.add_layer(layer[0], layer[1])
        return builder.build()


class NetworkLSTM(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        memory_size: int,
        output_layer: Tuple[int, Callable[[torch.Tensor], torch.Tensor]],
        *,
        hidden_sizes=[],
        device=None
    ):
        super(NetworkLSTM, self).__init__()
        self.device = device

        self.cell = CellLSTM(input_size, memory_size, hidden_sizes)
        self.output_net = (
            NetworkFF.Builder(memory_size)
            .add_layer(output_layer[0], output_layer[1])
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
