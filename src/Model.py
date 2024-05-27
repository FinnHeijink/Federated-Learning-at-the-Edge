import math

import torch
import torch.nn as nn
from torch.nn.functional import relu, max_pool2d, log_softmax, nll_loss, normalize, relu6, adaptive_avg_pool2d
from itertools import chain

class MLP(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, batchNorm, dtype):
        super(MLP, self).__init__()
        self.hiddenLayer = nn.Linear(inputSize, hiddenSize, bias=True, dtype=dtype)
        self.outputLayer = nn.Linear(hiddenSize, outputSize, bias=False, dtype=dtype)
        self.batchNorm = nn.BatchNorm1d(hiddenSize, **batchNorm)

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize

    def forward(self, x):
        return self.outputLayer(relu(self.batchNorm(self.hiddenLayer(x)), inplace=True))

    def getOutputSize(self):
        return self.outputSize

    def getInputSize(self):
        return self.inputSize

    def getHiddenSize(self):
        return self.hiddenSize

class Classifier(nn.Module):
    def __init__(self, classCount, hiddenSize, encoder, encoderName, batchNorm, dtypeName):
        super(Classifier, self).__init__()

        dtype = getattr(torch, dtypeName)

        self.encoder = globals()[encoderName](dtype=dtype, batchConfig=batchNorm, **encoder)
        self.outputLayer = MLP(self.encoder.getOutputSize(), hiddenSize, classCount, batchNorm=batchNorm, dtype=dtype)
        #self.outputLayer = nn.Linear(self.encoder.getOutputSize(), classCount, dtype=dtype)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            encoded = self.encoder(x).detach()

        return log_softmax(self.outputLayer(encoded), dim=1)

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
        return self.outputLayer.parameters()

class GenericEncoder(nn.Module):
    def __init__(self, imageDims, imageChannels, batchConfig, dtype, channels):
        super(GenericEncoder, self).__init__()

        self.imageDims = imageDims
        self.channels = channels

        sequence = []
        lastChannelCount = imageChannels
        for channel in channels:
            sequence.append(nn.Conv2d(lastChannelCount, channel, 3, 1, dtype=dtype))
            lastChannelCount = channel
            sequence.append(nn.ReLU())
        self.sequence = nn.Sequential(*sequence)

    def getOutputSize(self):
        return self.channels[-1] * (self.imageDims[0] - 2 * len(self.channels)) // 2 * (self.imageDims[1] - 2 * len(self.channels)) // 2

    def forward(self, x):
        x = self.sequence(x)
        x = max_pool2d(x, 2)
        return torch.flatten(x, 1)

class Encoder(GenericEncoder):
    def __init__(self, imageDims, imageChannels, batchConfig, dtype, outputChannels=64, hiddenChannels=32):
        super(Encoder, self).__init__(imageDims, imageChannels, batchConfig, dtype, channels=[hiddenChannels, outputChannels])

class EncoderType1(GenericEncoder):
    def __init__(self, imageDims, imageChannels, batchConfig, dtype):
        super(EncoderType1, self).__init__(imageDims, imageChannels, batchConfig, dtype, channels=[2, 4])

class EncoderType2(GenericEncoder):
    def __init__(self, imageDims, imageChannels, batchConfig, dtype):
        super(EncoderType2, self).__init__(imageDims, imageChannels, batchConfig, dtype, channels=[4, 8])

class EncoderType3(GenericEncoder):
    def __init__(self, imageDims, imageChannels, batchConfig, dtype):
        super(EncoderType3, self).__init__(imageDims, imageChannels, batchConfig, dtype, channels=[4, 8, 12])

class EncoderType4(GenericEncoder):
    def __init__(self, imageDims, imageChannels, batchConfig, dtype):
        super(EncoderType4, self).__init__(imageDims, imageChannels, batchConfig, dtype, channels=[6, 12, 18, 24, 30])

class MobileNetV2Block(nn.Module):
    def __init__(self, inputChannels, outputChannels, batchConfig, dtype, expansionFactor=6, downSample=False):
        super(MobileNetV2Block, self).__init__()

        self.downSample = downSample
        self.shortcut = (not downSample) and (inputChannels == outputChannels)

        internalChannels = inputChannels * expansionFactor

        self.conv1 = nn.Conv2d(inputChannels, internalChannels, 1, bias=False, dtype=dtype)
        self.bn1 = nn.BatchNorm2d(internalChannels, **batchConfig, dtype=dtype)
        self.conv2 = nn.Conv2d(internalChannels, internalChannels, 3, stride=2 if downSample else 1, groups=internalChannels, bias=False, padding=1, dtype=dtype)
        self.bn2 = nn.BatchNorm2d(internalChannels, **batchConfig, dtype=dtype)
        self.conv3 = nn.Conv2d(internalChannels, outputChannels, 1, bias=False, dtype=dtype)
        self.bn3 = nn.BatchNorm2d(outputChannels, **batchConfig, dtype=dtype)

    def forward(self, x):
        y = relu6(self.bn1(self.conv1(x)), inplace=True)
        y = relu6(self.bn2(self.conv2(y)), inplace=True)
        y = self.bn3(self.conv3(y))

        if self.shortcut:
            return y + x
        else:
            return y

class MobileNetV2(nn.Module):
    def __init__(self, dtype, imageDims, imageChannels, batchConfig):
        super(MobileNetV2, self).__init__()

        self.conv0 = nn.Conv2d(imageChannels, 32, 3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(32)

        self.blocks = nn.Sequential(
            MobileNetV2Block(32, 16, batchConfig, dtype, expansionFactor=1, downSample=False),
            MobileNetV2Block(16, 24, batchConfig, dtype, downSample=False),
            MobileNetV2Block(24, 24, batchConfig, dtype),
            MobileNetV2Block(24, 32, batchConfig, dtype, downSample=False),
            MobileNetV2Block(32, 32, batchConfig, dtype),
            MobileNetV2Block(32, 32, batchConfig, dtype),
            MobileNetV2Block(32, 64, batchConfig, dtype, downSample=True),
            MobileNetV2Block(64, 64, batchConfig, dtype),
            MobileNetV2Block(64, 64, batchConfig, dtype),
            MobileNetV2Block(64, 64, batchConfig, dtype),
            MobileNetV2Block(64, 96, batchConfig, dtype, downSample=False),
            MobileNetV2Block(96, 96, batchConfig, dtype),
            MobileNetV2Block(96, 96, batchConfig, dtype),
            MobileNetV2Block(96, 160, batchConfig, dtype, downSample=True),
            MobileNetV2Block(160, 160, batchConfig, dtype),
            MobileNetV2Block(160, 160, batchConfig, dtype),
            MobileNetV2Block(160, 320, batchConfig, dtype, downSample=False))

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
    def __init__(self, imageDims, imageChannels, batchConfig, dtype):
        super(MobileNetV2Short, self).__init__()

        self.conv0 = nn.Conv2d(imageChannels, 32, 3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(32)

        self.blocks = nn.Sequential(
            MobileNetV2Block(32, 16, batchConfig, dtype, expansionFactor=1, downSample=False),
            MobileNetV2Block(16, 24, batchConfig, dtype, downSample=False),
            #MobileNetV2Block(24, 24, batchConfig, dtype),
            MobileNetV2Block(24, 32, batchConfig, dtype, downSample=False),
            #MobileNetV2Block(32, 32, batchConfig, dtype),
            #MobileNetV2Block(32, 32, batchConfig, dtype),
            MobileNetV2Block(32, 64, batchConfig, dtype, downSample=True),
            #MobileNetV2Block(64, 64, batchConfig, dtype),
            #MobileNetV2Block(64, 64, batchConfig, dtype),
            #MobileNetV2Block(64, 64, batchConfig, dtype),
            MobileNetV2Block(64, 96, batchConfig, dtype, downSample=False),
            #MobileNetV2Block(96, 96, batchConfig, dtype),
            #MobileNetV2Block(96, 96, batchConfig, dtype),
            MobileNetV2Block(96, 160, batchConfig, dtype, downSample=True),
            #MobileNetV2Block(160, 160, batchConfig, dtype),
            #MobileNetV2Block(160, 160, batchConfig, dtype),
            MobileNetV2Block(160, 320, batchConfig, dtype, downSample=False))

        # last conv layers and fc layer
        self.conv1 = nn.Conv2d(320, 1280, 1, bias=False, dtype=dtype)
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
    def __init__(self, emaScheduler, encoderName, predictor, projector, encoder, batchNorm, dtypeName):
        super(BYOL, self).__init__()

        self.emaScheduler = emaScheduler

        dtype = getattr(torch, dtypeName)

        self.onlineEncoder = globals()[encoderName](dtype=dtype, batchConfig=batchNorm, **encoder)
        self.targetEncoder = globals()[encoderName](dtype=dtype, batchConfig=batchNorm, **encoder)
        self.onlineProjector = MLP(dtype=dtype, inputSize=self.onlineEncoder.getOutputSize(), batchNorm=batchNorm, **projector)
        self.targetProjector = MLP(dtype=dtype, inputSize=self.targetEncoder.getOutputSize(), batchNorm=batchNorm, **projector)
        self.predictor = MLP(dtype=dtype, inputSize=self.onlineProjector.getOutputSize(), outputSize=self.targetProjector.getOutputSize(), batchNorm=batchNorm, **predictor)

        # Make sure the target network starts out the same as the online network
        for onlineParam, targetParam in zip(self.onlineParameters(), self.targetParameters()):
            targetParam.requires_grad = False
            targetParam.data = onlineParam.data

    def forward(self, dataView1, dataView2):
        # dimensions of dataView1,2: [batchSize, channelCount, imageWidth, imageHeight]

        # Standard BYOL approach
        if (self.emaScheduler.getTau() != 0):
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
        # Simplified SimSiam approach
        else:
            image1OnlineEncoded = self.onlineEncoder(dataView1)
            image1Online = self.onlineProjector(image1OnlineEncoded)
            image1Predicted = self.predictor(image1Online)
            with torch.no_grad():
                image1Target = image1Online.detach()

            image2OnlineEncoded = self.onlineEncoder(dataView2)
            image2Online = self.onlineProjector(image2OnlineEncoded)
            image2Predicted = self.predictor(image2Online)
            with torch.no_grad():
                image2Target = image2Online.detach()
        def MSELoss(a, b):
            return torch.mean(torch.square(a - b))

        def RegressionLoss(a, b):
            return MSELoss(normalize(a, dim=1), normalize(b, dim=1))

        onlineLoss = (RegressionLoss(image1Predicted, image2Target) + RegressionLoss(image2Predicted, image1Target)) / 2

        if torch.isnan(onlineLoss).any():
            breakpoint()

        return onlineLoss

    def stepEMA(self):
        tau = self.emaScheduler.getTau()
        # Standard BYOL approach
        if (tau != 0):
            for onlineParam, targetParam in zip(self.onlineParameters(), self.targetParameters()):
                targetParam.data = targetParam.data + (onlineParam.data - targetParam.data) * (1.0 - tau)
        # Simplified SimSiam approach
        else:
            for onlineParam, targetParam in zip(self.onlineParameters(), self.targetParameters()):
                targetParam.data = onlineParam.data

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