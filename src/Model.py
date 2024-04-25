import torch
import torch.nn as nn
from torch.nn.functional import relu, max_pool2d, log_softmax, nll_loss
from itertools import chain

class MLP(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(MLP, self).__init__()
        self.hiddenLayer = nn.Linear(inputSize, hiddenSize, bias=True)
        self.outputLayer = nn.Linear(hiddenSize, outputSize, bias=False)

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize

    def forward(self, x):
        #Todo: batchnorm
        return self.outputLayer(relu(self.hiddenLayer(x)))

    def getOutputSize(self):
        return self.outputSize

    def getInputSize(self):
        return self.inputSize

    def getHiddenSize(self):
        return self.hiddenSize

class Predictor(MLP):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(Predictor, self).__init__(inputSize, hiddenSize, outputSize)

class Projector(MLP):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(Projector, self).__init__(inputSize, hiddenSize, outputSize)


class Classifier(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(Classifier, self).__init__()
        self.mlp = MLP(inputSize, hiddenSize, outputSize)

    def forward(self, x):
        return log_softmax(self.mlp(x), dim=1)

class Encoder(nn.Module):
    def __init__(self, imageDims, imageChannels, outputChannels=64, hiddenChannels=32, kernelSize=3):
        super(Encoder, self).__init__()

        self.imageDims = imageDims
        self.outputChannels = outputChannels

        self.conv1 = nn.Conv2d(imageChannels, hiddenChannels, kernelSize, 1)
        self.conv2 = nn.Conv2d(hiddenChannels, outputChannels, kernelSize, 1)
        self.dropout = nn.Dropout(0.25)

    def getOutputSize(self):
        return self.outputChannels * (self.imageDims[0] - 4) // 2 * (self.imageDims[1] - 4) // 2 #-4 due to the two 3x3 kernels, / 2 due to the pooling

    def forward(self, x):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = self.dropout(max_pool2d(x, 2))
        return torch.flatten(x, 1)

class EMA:
    def __init__(self, initialTau):
        self.tau = initialTau

    def apply(self, online, target):
        for onlineParam, targetParam in zip(online, target):
            targetParam = targetParam * self.tau + onlineParam * (1 - self.tau)

class BYOL(nn.Module):
    def __init__(self, ema, classCount, predictor, projector, classifier, encoder):
        super(BYOL, self).__init__()

        self.ema = ema

        self.onlineEncoder = Encoder(**encoder)
        self.targetEncoder = Encoder(**encoder)
        self.onlineProjector = Projector(inputSize=self.onlineEncoder.getOutputSize(), **projector)
        self.targetProjector = Projector(inputSize=self.targetEncoder.getOutputSize(), **projector)
        self.predictor = Predictor(inputSize=self.onlineProjector.getOutputSize(), outputSize=self.targetProjector.getOutputSize(), **predictor)
        self.classifier = Classifier(inputSize=self.onlineEncoder.getOutputSize(), outputSize=classCount, **classifier)

    def eval(self, x, target):
        output = self.classifier(self.onlineEncoder(x))
        prediction = output.argmax(dim=1, keepdim=True)
        loss = nll_loss(output, target)
        return output, prediction, loss

    def predict(self, x):
        output = self.classifier(self.onlineEncoder(x))
        prediction = output.argmax(dim=1, keepdim=True)
        return output, prediction

    def forward(self, dataView1, dataView2, target):
        # dimensions of dataView1,2: [batchSize, channelCount, imageWidth, imageHeight]

        image1OnlineEncoded = self.onlineEncoder(dataView1)
        image1Online = self.predictor(self.onlineProjector(image1OnlineEncoded))
        with torch.no_grad():
            image1Target = self.targetProjector(self.targetEncoder(dataView1))
        image1Classified = self.classifier(image1OnlineEncoded)

        image2OnlineEncoded = self.onlineEncoder(dataView2)
        image2Online = self.predictor(self.onlineProjector(image2OnlineEncoded))
        with torch.no_grad():
            image2Target = self.targetProjector(self.targetEncoder(dataView2))
        image2Classified = self.classifier(image2OnlineEncoded)

        classificationLoss = (nll_loss(image1Classified, target) + nll_loss(image2Classified, target)) / 2

        criterion = nn.MSELoss()
        onlineLoss = (criterion(image1Online, image2Target) + criterion(image2Online, image1Target)) / 2

        loss = onlineLoss + classificationLoss

        return loss

    def stepEMA(self):
        self.ema.apply(self.onlineParameters(), self.targetParameters())

    def trainableParameters(self):
        return chain(
            self.onlineEncoder.parameters(),
            self.onlineProjector.parameters(),
            self.predictor.parameters(),
            self.classifier.parameters()
        )

    def onlineParameters(self):
        return chain(
            self.onlineEncoder.parameters(),
            self.onlineProjector.parameters()
        )

    def targetParameters(self):
        return chain(
            self.targetEncoder.parameters(),
            self.targetProjector.parameters()
        )