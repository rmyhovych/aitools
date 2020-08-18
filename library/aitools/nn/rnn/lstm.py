import torch
from torch.nn import Linear as L


class CellLSTM(torch.jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(CellLSTM, self).__init__()
        self.hidden_size = hidden_size

        interm_size = input_size + hidden_size
        self.forget_layer = L(interm_size, hidden_size)

        self.update_layer_force = L(interm_size, hidden_size)
        self.update_layer_direction = L(interm_size, hidden_size)

        self.output_layer = L(interm_size, hidden_size)

    @torch.jit.script_method
    def forward(self, x, h, state):
        xh = torch.cat((x, h), dim=0)

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
        self.output_layer = L(hidden_size, output_size)

        self.output_activation = (
            torch.tensor if output_activation is None else output_activation
        )

        self.to(self.device)

    @torch.jit.script_method
    def forward(self, xs):
        h, state = (
            torch.zeros((self.cell.hidden_size,), device=self.device),
            torch.zeros((self.cell.hidden_size,), device=self.device),
        )

        for x in xs:
            h, state = self.cell(x, h, state)

        return self.output_activation(self.output_layer(h))
