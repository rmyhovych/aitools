import torch
import torch.nn as nn


class NetworkLSTM(nn.Module):
    def __init__(self, inSize, hiddenSize, outSize, outAct):
        super(NetworkLSTM, self).__init__()
        self.parameterList = nn.ModuleList()

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
        layer = nn.Linear(inSize, outSize, bias=True)
        self.parameterList.append(layer)

        def forgetGate(xh):
            return torch.sigmoid(layer(xh))

        return forgetGate

    def _createUpdateGate(self, inSize, outSize):
        layerInput = nn.Linear(inSize, outSize, bias=True)
        self.parameterList.append(layerInput)

        layerAdd = nn.Linear(inSize, outSize, bias=True)
        self.parameterList.append(layerAdd)

        def updateGate(xh):
            outInput = torch.sigmoid(layerInput(xh))
            outAdd = torch.tanh(layerAdd(xh))

            return torch.mul(outInput, outAdd)

        return updateGate

    def _createOutputGate(self, inSize, outSize):
        layerOutput = nn.Linear(inSize, outSize, bias=True)
        self.parameterList.append(layerOutput)

        def outputGate(xh, cellState):
            out = torch.sigmoid(layerOutput(xh))
            cellStateTanh = torch.tanh(cellState)

            return torch.mul(cellStateTanh, out)

        return outputGate

    def _createOutputLayer(self, inSize, outSize, outAct):
        layer = nn.Linear(inSize, outSize, bias=True)
        self.parameterList.append(layer)

        if outAct is None:
            return lambda hidden: layer(hidden)
        else:
            return lambda hidden: outAct(layer(hidden))


class NetworkBDLSTMFull(nn.Module):
    def __init__(self, inSize, hiddenSize, outSize, outAct):
        super(NetworkBDLSTMFull, self).__init__()

        self.netForward = NetworkLSTM(inSize, hiddenSize, outSize, None)
        self.netBackward = NetworkLSTM(inSize, hiddenSize, outSize, None)

        self.outLayer = nn.Linear(2 * outSize, outSize, bias=True)
        self.outAct = outAct

        self.parameterList = nn.ModuleList(
            [self.netForward, self.netBackward, self.outLayer]
        )

    def reset(self):
        self.forwardN.reset()
        self.backwardN.reset()

    def forward(self, sentence):
        sentenceBackward = torch.flip(sentence, (0,))

        self._feedNoOut(self.netForward, sentence[:-1])
        self._feedNoOut(self.netBackward, sentenceBackward[:-1])

        outForward = self.netForward(sentence[-1])
        outBackward = self.netBackward(sentenceBackward[-1])

        y = torch.cat([outForward, outBackward], dim=0)
        y = self.outLayer(y)
        if self.outAct is not None:
            y = self.outAct(y)

        return y

    def _feedNoOut(self, net, data):
        net.reset()
        for character in data:
            net(character)


class NetworkBDLSTMCrop(NetworkBDLSTMFull):
    def forward(self, sentence, index):
        beginning = sentence[:index]
        end = torch.flip(sentence[index:], (0,))

        self._feedNoOut(self.netForward, beginning[:-1])
        self._feedNoOut(self.netBackward, end[:-1])

        outForward = self.netForward(beginning[-1])
        outBackward = self.netBackward(end[-1])

        y = torch.cat([outForward, outBackward], dim=0)
        y = self.outLayer(y)
        if self.outAct is not None:
            y = self.outAct(y)

        return y
