import torch

from .lstm import NetworkLSTM


class NetworkLSTMBD(torch.nn.Module):
    def __init__(self, inSize, hiddenSize, outSize, outAct):
        super(NetworkLSTMBD, self).__init__()

        self.netForward = NetworkLSTM(inSize, hiddenSize, outSize, None)
        self.netBackward = NetworkLSTM(inSize, hiddenSize, outSize, None)

        self.outLayer = torch.nn.Linear(2 * outSize, outSize, bias=True)
        self.outAct = outAct

        self.parameterList = torch.nn.ModuleList(
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


class NetworkLSTMBDCrop(NetworkLSTMBD):
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
