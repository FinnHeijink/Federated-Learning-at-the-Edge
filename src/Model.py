import torch
import torch.nn as nn
from torch.nn.functional import relu, max_pool2d, log_softmax, nll_loss

class MLP(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(MLP, self).__init__()
        self.hiddenLayer = nn.Linear(inputSize, hiddenSize, bias=True)
        self.outputLayer = nn.Linear(hiddenSize, outputSize, bias=False)

    def forward(self, x):
        #Todo: batchnorm
        return self.outputLayer(relu(self.inputLayer(x)))

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
        return log_softmax(self.mlp(x), dim=1)

class Encoder(nn.Module):
    def __init__(self, imageDims, imageChannels):
        super(Encoder, self).__init__()

        self.imageDims = imageDims

        self.conv1 = nn.Conv2d(imageChannels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout = nn.Dropout(0.25)

    def getOutputSize(self):
        return (self.imageDims[0] - 4) // 2 * (self.imageDims[0] - 4) // 2 #-4 due to the two 3x3 kernels, / 2 due to the pooling

    def forward(self, x):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = self.dropout(max_pool2d(x, 2))
        return torch.flatten(x, 1)

class BYOL(nn.Module):
    def __init__(self, classCount, predictor, projector, classifier, encoder):
        super(BYOL, self).__init__()

        self.onlineEncoder = Encoder(**encoder)
        self.targetEncoder = Encoder(**encoder)
        self.onlineProjector = Projector(inputSize=self.onlineEncoder.getOutputSize(), **projector)
        self.targetProjector = Projector(inputSize=self.targetEncoder.getOutputSize(), **projector)
        self.predictor = Predictor(inputSize=self.onlineProjector.getOutputSize(), outputSize=self.targetProjector.getOutputSize(), **predictor)
        self.classifier = Classifier(inputSize=self.onlineEncoder.getOutputSize(), outputSize=classCount, **classifier)

    def forward(self, x, target):
        # dimensions of x: [batchSize, imageViews=2, channelCount, imageWidth, imageHeight]
        image1, image2 = torch.split(x, split_size_or_sections=1, dim=1)
        image1, image2 = torch.squeeze(image1, dim=1), torch.squeeze(image2, dim=1)

        image1OnlineEncoded = self.onlineEncoder(image1)
        image1Online = self.predictor(self.onlineProjector(image1OnlineEncoded))
        image1Target = self.targetProjector(self.targetEncoder(image1))
        image1Classified = self.classifier(image1OnlineEncoded)

        image2OnlineEncoded = self.onlineEncoder(image2)
        image2Online = self.predictor(self.onlineProjector(image2OnlineEncoded))
        image2Target = self.targetProjector(self.targetEncoder(image2))
        image2Classified = self.classifier(image2OnlineEncoded)

        classificationLoss = (nll_loss(image1Classified, target) + nll_loss(image2Classified, target)) / 2

        criterion = nn.MSELoss()
        onlineLoss = (criterion(image1Online, image2Target) + criterion(image2Online, image1Target)) / 2

        loss = onlineLoss + classificationLoss

        return loss

    def trainableParameters(self):
        raise NotImplementedError #Should only return the parameters of the online part of the network