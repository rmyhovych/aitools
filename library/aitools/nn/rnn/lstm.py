import torch


class NetworkLSTM(torch.nn.Module):
    def __init__(self, inSize, hiddenSize, outSize, outAct):
        super(NetworkLSTM, self).__init__()
        self.parameterList = torch.nn.ModuleList()

        catInSize = inSize + hiddenSize
        self.forgetGate = self._createForgetGate(catInSize, hiddenSize)
        self.updateGate = self._createUpdateGate(catInSize, hiddenSize)
        self.outputGate = self._createOutputGate(catInSize, hiddenSize)
        self.outputLayer = self._createOutputLayer(hiddenSize, outSize, outAct)

        self.hidden = torch.zeros(size=(hiddenSize,))
        self.cellState = torch.zeros(size=(hiddenSize,))

    def reset(self):
        self.hidden = torch.zeros(size=self.hidden.size())
        self.cellState = torch.zeros(size=self.cellState.size())

    def forward(self, x):
        xh = torch.cat((x, self.hidden), dim=0)

        forget = self.forgetGate(xh)
        self.cellState = torch.mul(forget, self.cellState)

        update = self.updateGate(xh)
        self.cellState = torch.add(update, self.cellState)

        self.hidden = self.outputGate(xh, self.cellState)

        return self.outputLayer(self.hidden)

    def _createForgetGate(self, inSize, outSize):
        layer = torch.nn.Linear(inSize, outSize, bias=True)
        self.parameterList.append(layer)

        def forgetGate(xh):
            return torch.sigmoid(layer(xh))

        return forgetGate

    def _createUpdateGate(self, inSize, outSize):
        layerInput = torch.nn.Linear(inSize, outSize, bias=True)
        self.parameterList.append(layerInput)

        layerAdd = torch.nn.Linear(inSize, outSize, bias=True)
        self.parameterList.append(layerAdd)

        def updateGate(xh):
            outInput = torch.sigmoid(layerInput(xh))
            outAdd = torch.tanh(layerAdd(xh))

            return torch.mul(outInput, outAdd)

        return updateGate

    def _createOutputGate(self, inSize, outSize):
        layerOutput = torch.nn.Linear(inSize, outSize, bias=True)
        self.parameterList.append(layerOutput)

        def outputGate(xh, cellState):
            out = torch.sigmoid(layerOutput(xh))
            cellStateTanh = torch.tanh(cellState)

            return torch.mul(cellStateTanh, out)

        return outputGate

    def _createOutputLayer(self, inSize, outSize, outAct):
        layer = torch.nn.Linear(inSize, outSize, bias=True)
        self.parameterList.append(layer)

        if outAct is None:
            return lambda hidden: layer(hidden)
        else:
            return lambda hidden: outAct(layer(hidden))
