import torch
import torch.nn as nn
from torch.nn.functional import relu, max_pool2d, log_softmax, nll_loss, normalize, relu6, adaptive_avg_pool2d
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
    def __init__(self, classCount, hiddenSize, encoder, encoderName, batchNorm):
        super(Classifier, self).__init__()

        self.encoder = globals()[encoderName](**encoder)
        self.fc1 = nn.Linear(self.encoder.getOutputSize(), hiddenSize)
        self.bn1 = nn.BatchNorm1d(hiddenSize, **batchNorm)
        self.fc2 = nn.Linear(hiddenSize, classCount)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            encoded = self.encoder(x).detach()

        return log_softmax(self.fc2(relu(self.bn1(self.fc1(encoded)))), dim=1)

    def loss(self, x, target):
        return nll_loss(self(x), target)

    def predict(self, x):
        output = self(x)
        prediction = output.argmax(dim=1, keepdim=True)
        return output, prediction

    def predictionLoss(self, x, target):
        output = self(x)
        prediction = output.argmax(dim=1, keepdim=True)
        loss = nll_loss(output, target)
        return loss, output, prediction

    def copyEncoderFromBYOL(self, byol):
        for classifierParam, onlineParam in zip(self.encoder.parameters(), byol.onlineEncoderParameters()):
            classifierParam.data = onlineParam.data

    def trainableParameters(self):
        return chain(
            self.fc1.parameters(),
            self.bn1.parameters(),
            self.fc2.parameters()
        )
        #return self.fc.parameters() # Todo: what if we add an another layer? Automate

class Encoder(nn.Module):
    def __init__(self, imageDims, imageChannels, outputChannels=64, hiddenChannels=32, kernelSize=3):
        super(Encoder, self).__init__()

        self.imageDims = imageDims
        self.outputChannels = outputChannels

        self.conv1 = nn.Conv2d(imageChannels, hiddenChannels, kernelSize, 1)
        self.conv2 = nn.Conv2d(hiddenChannels, outputChannels, kernelSize, 1)
        self.dropout = nn.Dropout(0.2)

    def getOutputSize(self):
        return self.outputChannels * (self.imageDims[0] - 4) // 2 * (self.imageDims[1] - 4) // 2 #-4 due to the two 3x3 kernels, / 2 due to the pooling

    def forward(self, x):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = self.dropout(max_pool2d(x, 2))
        return torch.flatten(x, 1)

class MobileNetV2Block(nn.Module):
    def __init__(self, inputChannels, outputChannels, expansionFactor=6, downSample=False):
        super(MobileNetV2Block, self).__init__()

        self.downSample = downSample
        self.shortcut = (not downSample) and (inputChannels == outputChannels)

        internalChannels = inputChannels * expansionFactor

        self.conv1 = nn.Conv2d(inputChannels, internalChannels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(internalChannels)
        self.conv2 = nn.Conv2d(internalChannels, internalChannels, 3, stride=2 if downSample else 1, groups=internalChannels, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(internalChannels)
        self.conv3 = nn.Conv2d(internalChannels, outputChannels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(outputChannels)

    def forward(self, x):
        y = relu6(self.bn1(self.conv1(x)))
        y = relu6(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))

        if self.shortcut:
            return y + x
        else:
            return y

class MobileNetV2(nn.Module):
    def __init__(self, imageDims, imageChannels):
        super(MobileNetV2, self).__init__()

        self.conv0 = nn.Conv2d(imageChannels, 32, 3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(32)

        self.blocks = nn.Sequential(
            MobileNetV2Block(32, 16, expansionFactor=1, downSample=False),
            MobileNetV2Block(16, 24, downSample=False),
            MobileNetV2Block(24, 24),
            MobileNetV2Block(24, 32, downSample=False),
            MobileNetV2Block(32, 32),
            MobileNetV2Block(32, 32),
            MobileNetV2Block(32, 64, downSample=True),
            MobileNetV2Block(64, 64),
            MobileNetV2Block(64, 64),
            MobileNetV2Block(64, 64),
            MobileNetV2Block(64, 96, downSample=False),
            MobileNetV2Block(96, 96),
            MobileNetV2Block(96, 96),
            MobileNetV2Block(96, 160, downSample=True),
            MobileNetV2Block(160, 160),
            MobileNetV2Block(160, 160),
            MobileNetV2Block(160, 320, downSample=False))

        # last conv layers and fc layer
        self.conv1 = nn.Conv2d(320, 1280, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(1280)

    def getOutputSize(self):
        return 1280

    def forward(self, x):
        y = relu6(self.bn0(self.conv0(x)))
        y = self.blocks(y)
        y = relu6(self.bn1(self.conv1(y)))
        y = adaptive_avg_pool2d(y, 1)
        y = torch.squeeze(torch.squeeze(y, -1), -1)
        return y

class MobileNetV2Short(nn.Module):
    def __init__(self, imageDims, imageChannels):
        super(MobileNetV2Short, self).__init__()

        self.conv0 = nn.Conv2d(imageChannels, 32, 3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(32)

        self.blocks = nn.Sequential(
            MobileNetV2Block(32, 16, expansionFactor=1, downSample=False),
            MobileNetV2Block(16, 24, downSample=False),
            #MobileNetV2Block(24, 24),
            MobileNetV2Block(24, 32, downSample=False),
            #MobileNetV2Block(32, 32),
            #MobileNetV2Block(32, 32),
            MobileNetV2Block(32, 64, downSample=True),
            #MobileNetV2Block(64, 64),
            #MobileNetV2Block(64, 64),
            #MobileNetV2Block(64, 64),
            MobileNetV2Block(64, 96, downSample=False),
            #MobileNetV2Block(96, 96),
            #MobileNetV2Block(96, 96),
            MobileNetV2Block(96, 160, downSample=True),
            #MobileNetV2Block(160, 160),
            #MobileNetV2Block(160, 160),
            MobileNetV2Block(160, 320, downSample=False))

        # last conv layers and fc layer
        self.conv1 = nn.Conv2d(320, 1280, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(1280)

    def getOutputSize(self):
        return 1280

    def forward(self, x):
        y = relu6(self.bn0(self.conv0(x)))
        y = self.blocks(y)
        y = relu6(self.bn1(self.conv1(y)))
        y = adaptive_avg_pool2d(y, 1)
        y = torch.squeeze(torch.squeeze(y, -1), -1)
        return y

class BYOL(nn.Module):
    def __init__(self, initialTau, encoderName, predictor, projector, encoder, batchNorm):
        super(BYOL, self).__init__()

        self.tau = initialTau

        self.onlineEncoder = globals()[encoderName](**encoder)
        self.targetEncoder = globals()[encoderName](**encoder)
        self.onlineProjector = MLP(inputSize=self.onlineEncoder.getOutputSize(), batchNorm=batchNorm, **projector)
        self.targetProjector = MLP(inputSize=self.targetEncoder.getOutputSize(), batchNorm=batchNorm, **projector)
        self.predictor = MLP(inputSize=self.onlineProjector.getOutputSize(), outputSize=self.targetProjector.getOutputSize(), batchNorm=batchNorm, **predictor)

        # Make sure the target network starts out the same as the online network
        for onlineParam, targetParam in zip(self.onlineParameters(), self.targetParameters()):
            targetParam.requires_grad = False
            targetParam.data = onlineParam.data

    def forward(self, dataView1, dataView2):
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
        for onlineParam, targetParam in zip(self.onlineParameters(), self.targetParameters()):
            targetParam.data = targetParam.data + (onlineParam.data - targetParam.data) * (1.0 - self.tau)

    def trainableParameters(self):
        return chain(
            self.onlineEncoder.parameters(),
            self.onlineProjector.parameters(),
            self.predictor.parameters(),
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

    def onlineEncoderParameters(self):
        return self.onlineEncoder.parameters()

    def predictorParameters(self):
        return self.predictor.parameters()