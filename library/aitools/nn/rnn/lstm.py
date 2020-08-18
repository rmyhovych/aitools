import torch
from torch.nn import Bilinear
from torch.nn import Linear


class CellLSTM(torch.jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(CellLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.forget_layer = Bilinear(input_size, hidden_size, hidden_size)

        self.update_layer_force = Bilinear(input_size, hidden_size, hidden_size)
        self.update_layer_direction = Bilinear(input_size, hidden_size, hidden_size)

        self.output_layer = Bilinear(input_size, hidden_size, hidden_size)

    @torch.jit.script_method
    def forward(self, x, h, state):
        new_state = torch.mul(self._forget(x, h), state)
        new_state = torch.add(self._update(x, h), new_state)
        new_h = torch.mul(self._output(x, h), torch.tanh(new_state))

        return new_h, new_state

    def _forget(self, x, h):
        return torch.sigmoid(self.forget_layer(x, h))

    def _update(self, x, h):
        force = torch.sigmoid(self.update_layer_force(x, h))
        direction = torch.tanh(self.update_layer_direction(x, h))
        return torch.mul(force, direction)

    def _output(self, x, h):
        return torch.sigmoid(self.output_layer(x, h))


class NetworkLSTM(torch.jit.ScriptModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        output_activation=None,
        *,
        device=None
    ):
        super(NetworkLSTM, self).__init__()
        self.device = device

        self.cell = CellLSTM(input_size, hidden_size)

        self.output_layer = Linear(hidden_size, output_size)
        self.output_activation = output_activation

        self.to(self.device)

    @torch.jit.script_method
    def forward(self, xs):
        h, state = (
            torch.zeros((self.cell.hidden_size,), device=self.device),
            torch.zeros((self.cell.hidden_size,), device=self.device),
        )

        for x in xs:
            h, state = self.cell(x, h, state)

        return self._output(h)

    @torch.jit.script_method
    def _output(self, h):
        y = self.output_layer(h)
        if self.output_activation is None:
            return y
        else:
            return self.output_activation(y)
