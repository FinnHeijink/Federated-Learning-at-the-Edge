import torch
import torch.nn as nn
from torch.nn.functional import relu, max_pool2d, log_softmax, nll_loss

class MLP(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(MLP, self).__init__()

    def forward(self, x):
        raise NotImplementedError
        return x

class Predictor(MLP):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(Predictor, self).__init__(inputSize, hiddenSize, outputSize)

class Projector(MLP):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(Projector, self).__init__(inputSize, hiddenSize, outputSize)

        self.outputSize = outputSize

    def getOutputSize(self):
        return self.outputSize


class Classifier(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(Classifier, self).__init__()
        self.mlp = MLP(inputSize, hiddenSize, outputSize)

    def forward(self, x):
        raise NotImplementedError
        return x

class Encoder(nn.Module):
    def __init__(self, imageDims):
        super(Encoder, self).__init__()

    def getOutputSize(self):
        raise NotImplementedError
        return 0

    def forward(self, x):
        raise NotImplementedError
        return x

class BYOL(nn.Module):
    def __init__(self, classCount, predictor, projector, classifier, encoder):
        super(BYOL, self).__init__()

        self.onlineEncoder = Encoder(**encoder)
        self.targetEncoder = Encoder(**encoder)
        self.onlineProjector = Projector(inputSize=self.onlineEncoder.getOutputSize(), **projector)
        self.targetProjector = Projector(inputSize=self.targetEncoder.getOutputSize(), **projector)
        self.predictor = Predictor(inputSize=self.onlineProjector.getOutputSize(), outputSize=self.targetProjector.getOutputSize(), **predictor)
        self.classifier = Classifier(inputSize=self.onlineEncoder.getOutputSize(), outputSize=classCount, **classifier)

    def forward(self, x):
        raise NotImplementedError
        return x

    def trainableParameters(self):
        raise NotImplementedError #Should only return the parameters of the online part of the network