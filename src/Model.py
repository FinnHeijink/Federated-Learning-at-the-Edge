import torch
import torch.nn as nn
from torch.nn.functional import relu, max_pool2d, log_softmax, nll_loss, normalize
from itertools import chain

class MLP(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, batchNorm):
        super(MLP, self).__init__()
        self.hiddenLayer = nn.Linear(inputSize, hiddenSize, bias=True)
        self.outputLayer = nn.Linear(hiddenSize, outputSize, bias=False)
        self.batchNorm = nn.BatchNorm1d(hiddenSize, **batchNorm)

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize

    def forward(self, x):
        #Todo: batchnorm
        return self.outputLayer(relu(self.batchNorm(self.hiddenLayer(x))))

    def getOutputSize(self):
        return self.outputSize

    def getInputSize(self):
        return self.inputSize

    def getHiddenSize(self):
        return self.hiddenSize

class Classifier(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, batchNorm):
        super(Classifier, self).__init__()
        self.mlp = MLP(inputSize, hiddenSize, outputSize, batchNorm)

    def forward(self, x):
        return log_softmax(self.mlp(x), dim=1)


class Classifier2(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, batchNorm):
        super(Classifier2, self).__init__()

        self.fc = nn.Linear(inputSize, outputSize)

    def forward(self, x):
        return log_softmax(self.fc(x), dim=1)

class Encoder(nn.Module):
    def __init__(self, imageDims, imageChannels, outputChannels=64, hiddenChannels=32, kernelSize=3):
        super(Encoder, self).__init__()

        self.imageDims = imageDims
        self.outputChannels = outputChannels

        self.conv1 = nn.Conv2d(imageChannels, hiddenChannels, kernelSize, 1)
        self.conv2 = nn.Conv2d(hiddenChannels, outputChannels, kernelSize, 1)
        self.dropout = nn.Dropout(0)

        # self.fc = nn.Linear(self.getOutputSize(), self.getOutputSize())

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
            targetParam.data  = targetParam.data + (onlineParam.data - targetParam.data) * (1.0 - self.tau)

class BYOL(nn.Module):
    def __init__(self, ema, classCount, predictor, projector, classifier, encoder, batchNorm):
        super(BYOL, self).__init__()

        self.ema = ema

        self.onlineEncoder = Encoder(**encoder)
        self.targetEncoder = Encoder(**encoder)
        self.classifierEncoder = Encoder(**encoder)
        self.onlineProjector = MLP(inputSize=self.onlineEncoder.getOutputSize(), batchNorm=batchNorm, **projector)
        self.targetProjector = MLP(inputSize=self.targetEncoder.getOutputSize(), batchNorm=batchNorm, **projector)
        self.predictor = MLP(inputSize=self.onlineProjector.getOutputSize(), outputSize=self.targetProjector.getOutputSize(), batchNorm=batchNorm, **predictor)
        self.classifier = Classifier2(inputSize=self.onlineEncoder.getOutputSize(), outputSize=classCount, batchNorm=batchNorm, **classifier)

        # Make sure the target network starts out the same as the online network
        for onlineParam, targetParam in zip(self.onlineParameters(), self.targetParameters()):
            targetParam.requires_grad = False
            targetParam.data = onlineParam.data

        for classifierParam in self.classifierEncoder.parameters():
            classifierParam.requires_grad = False

    def predictionLoss(self, x, target):
        output = self.classifier(self.onlineEncoder(x))
        prediction = output.argmax(dim=1, keepdim=True)
        loss = nll_loss(output, target)
        return loss, output, prediction

    def predict(self, x):
        output = self.classifier(self.onlineEncoder(x))
        prediction = output.argmax(dim=1, keepdim=True)
        return output, prediction

    def classificationLoss(self, x, target):
        with torch.no_grad():
            encoded = self.classifierEncoder(x)

        output = self.classifier(encoded.detach())
        return nll_loss(output, target)

    def forward(self, dataView1, dataView2, target):
        # dimensions of dataView1,2: [batchSize, channelCount, imageWidth, imageHeight]

        image1OnlineEncoded = self.onlineEncoder(dataView1)
        image1Online = self.onlineProjector(image1OnlineEncoded)
        image1Predicted = self.predictor(image1Online)
        with torch.no_grad():
            image1Target = self.targetProjector(self.targetEncoder(dataView1)).detach()

        image2OnlineEncoded = self.onlineEncoder(dataView2)
        image2Online = self.onlineProjector(image2OnlineEncoded)
        image2Predicted = self.predictor(image2Online)
        with torch.no_grad():
            image2Target = self.targetProjector(self.targetEncoder(dataView2)).detach()

        def MSELoss(a, b):
            return torch.mean(torch.square(a - b))

        def RegressionLoss(a, b):
            return MSELoss(normalize(a, dim=1), normalize(b, dim=1))

        onlineLoss = (RegressionLoss(image1Predicted, image2Target) + RegressionLoss(image2Predicted, image1Target)) / 2

        return onlineLoss

    def stepEMA(self):
        self.ema.apply(self.onlineParameters(), self.targetParameters())

    def copyClassifierEncoder(self):
        for classifierParam, onlineParam in zip(self.classifierEncoder.parameters(), self.onlineEncoder.parameters()):
            classifierParam.data = onlineParam.data

    def trainableParameters(self):
        return chain(
            self.onlineEncoder.parameters(),
            self.onlineProjector.parameters(),
            self.predictor.parameters(),
            self.classifier.parameters()
        )

    def onlineParameters(self):
        #return self.onlineProjector.parameters()
        return chain(
            self.onlineEncoder.parameters(),
            self.onlineProjector.parameters()
        )

    def targetParameters(self):
        #return self.targetProjector.parameters()
        return chain(
            self.targetEncoder.parameters(),
            self.targetProjector.parameters()
        )