import torch
from torch.nn import Linear


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

    def forward(self, xs):
        h, state = self._create_new_states()

        for x in xs:
            h, state = self.cell(x, h, state)

        return self._output(h)

    def _output(self, h):
        y = self.output_layer(h)
        if self.output_activation is None:
            return y
        else:
            return self.output_activation(y)

    def _create_new_states(self):
        return (
            torch.zeros((self.cell.hidden_size,), device=self.device),
            torch.zeros((self.cell.hidden_size,), device=self.device),
        )
