import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import relu, max_pool2d, log_softmax, nll_loss, normalize, relu6, adaptive_avg_pool2d
from itertools import chain

import KRIAInterface
import QNN

def relu1(x, inplace=False):
    return torch.clamp(x, 0, 1)

class ReLU1(nn.Module):
    def __init__(self):
        super(ReLU1, self).__init__()

    def forward(self, x):
        return relu1(x)

reluFunctionToUse=relu
reluModuleToUse=nn.ReLU

def SetUseReLU1(use):
    global reluFunctionToUse
    global reluModuleToUse
    if use:
        reluFunctionToUse = relu1
        reluModuleToUse = ReLU1
    else:
        reluFunctionToUse = relu
        reluModuleToUse = nn.ReLU

def GetLinearComputeCost(inputSize, outputSize, bias):
    if bias:
        inputSize += 1

    multiplies = inputSize * outputSize
    adds = outputSize * (inputSize - 1)
    params = inputSize * outputSize
    return np.array((multiplies, adds, params))

def GetConvolutionalComputeCost(inputDims, inputChannels, outputChannels, kernelSize, stride=1):
    perPixel = np.array((kernelSize ** 2, kernelSize ** 2 + 1))

    pixels = (inputDims[0] - (kernelSize - 1))/stride * (inputDims[1] - (kernelSize - 1))/stride

    totalMA = perPixel * pixels * inputChannels * outputChannels
    params = kernelSize ** 2 * inputChannels * outputChannels
    return np.concatenate((totalMA, [params]))

def GetBatchNormComputeCost(size):
    return np.array((0, 0, 0))

class MLP(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, batchNorm, dtype, quantization):
        super(MLP, self).__init__()

        self.hiddenLayer = nn.Linear(inputSize, hiddenSize, bias=True, dtype=dtype)
        self.outputLayer = nn.Linear(hiddenSize, outputSize, bias=False, dtype=dtype)
        self.batchNorm = nn.BatchNorm1d(hiddenSize, **batchNorm)

        self.quantizationEnabled = quantization["enabled"]

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize

    def forward(self, x):
        x = self.hiddenLayer(x)
        if self.quantizationEnabled:
            x = QNN.quantize(x)
        x = self.batchNorm(x)
        if self.quantizationEnabled:
            x = QNN.quantize(x)
        x = reluFunctionToUse(x, inplace=True)
        x = self.outputLayer(x)
        if self.quantizationEnabled:
            x = QNN.quantize(x)
        return x
        #return self.outputLayer(reluFunctionToUse(self.hiddenLayer(x), inplace=True))

    def getOutputSize(self):
        return self.outputSize

    def getInputSize(self):
        return self.inputSize

    def getHiddenSize(self):
        return self.hiddenSize

    def getComputeCost(self):
        return GetLinearComputeCost(self.inputSize, self.hiddenSize, True) +\
            GetLinearComputeCost(self.hiddenSize, self.outputSize, False) +\
            GetBatchNormComputeCost(self.hiddenSize)

class Classifier(nn.Module):
    def __init__(self, classCount, hiddenSize, encoder, encoderName, batchNorm, dtypeName, quantization):
        super(Classifier, self).__init__()

        dtype = getattr(torch, dtypeName)

        QNN.QuantizeTensor.nb = quantization["nb"]
        QNN.QuantizeTensor.nf = quantization["nf"]
        self.quantizationEnabled = quantization["enabled"]

        self.encoder = globals()[encoderName](dtype=dtype, batchConfig=batchNorm, quantization=quantization, **encoder)
        self.outputLayer = MLP(self.encoder.getOutputSize(), hiddenSize, classCount, batchNorm=batchNorm, dtype=dtype, quantization=quantization)
        #self.outputLayer = nn.Linear(self.encoder.getOutputSize(), classCount, dtype=dtype)

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.allowTrainingEncoder = False

    def setAllowTrainingEncoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

        self.allowTrainingEncoder = True

    def forward(self, x):

        if self.allowTrainingEncoder:
            if self.quantizationEnabled:
                x = QNN.quantize(x)
            encoded = self.encoder(x)
        else:
            with torch.no_grad():
                if self.quantizationEnabled:
                    x = QNN.quantize(x)
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
        if self.allowTrainingEncoder:
            return self.parameters()
        else:
            return self.outputLayer.parameters()

    def getComputeCost(self):
        return self.encoder.getComputeCost() + self.outputLayer.getComputeCost()

class GenericEncoder(nn.Module):
    def __init__(self, imageDims, imageChannels, batchConfig, dtype, quantization, channels):
        super(GenericEncoder, self).__init__()

        self.imageDims = imageDims
        self.channels = channels

        self.quantizationEnabled = quantization["enabled"]
        self.useCustomConv = quantization["useCustomConv"]

        sequence = []
        lastChannelCount = imageChannels
        computeCost = np.zeros((3,))
        currentImageDims = list(imageDims)

        for channel in channels:
            if self.useCustomConv:
                sequence.append(KRIAInterface.Conv2D_3x3(lastChannelCount, channel, dtype=dtype))
            else:
                sequence.append(nn.Conv2d(lastChannelCount, channel, 3, 1, dtype=dtype))
            if self.quantizationEnabled:
                sequence.append(QNN.QuantizeModel())

            sequence.append(nn.BatchNorm2d(channel, **batchConfig))
            if self.quantizationEnabled:
                sequence.append(QNN.QuantizeModel())

            sequence.append(reluModuleToUse())

            computeCost += GetConvolutionalComputeCost(currentImageDims, lastChannelCount, channel, 3)

            currentImageDims[0] -= 1
            currentImageDims[1] -= 1
            lastChannelCount = channel

        self.computeCost = computeCost

        self.sequence = nn.Sequential(*sequence)

    def getOutputSize(self):
        return self.channels[-1] * (self.imageDims[0] - 2 * len(self.channels)) // 2 * (self.imageDims[1] - 2 * len(self.channels)) // 2

    def forward(self, x):
        x = self.sequence(x)
        x = max_pool2d(x, 2)
        return torch.flatten(x, 1)

    def getComputeCost(self):
        return self.computeCost

class Encoder(GenericEncoder):
    def __init__(self, imageDims, imageChannels, batchConfig, dtype, quantization, outputChannels=64, hiddenChannels=32):
        super(Encoder, self).__init__(imageDims, imageChannels, batchConfig, dtype, quantization, channels=[hiddenChannels, outputChannels])

class EncoderType1(GenericEncoder):
    def __init__(self, imageDims, imageChannels, batchConfig, dtype, quantization):
        super(EncoderType1, self).__init__(imageDims, imageChannels, batchConfig, dtype, quantization, channels=[2, 4])

class EncoderType2(GenericEncoder):
    def __init__(self, imageDims, imageChannels, batchConfig, dtype, quantization):
        super(EncoderType2, self).__init__(imageDims, imageChannels, batchConfig, dtype, quantization, channels=[4, 8])

class EncoderType3(GenericEncoder):
    def __init__(self, imageDims, imageChannels, batchConfig, dtype, quantization):
        super(EncoderType3, self).__init__(imageDims, imageChannels, batchConfig, dtype, quantization, channels=[4, 8, 12])

class EncoderType4(GenericEncoder):
    def __init__(self, imageDims, imageChannels, batchConfig, dtype, quantization):
        super(EncoderType4, self).__init__(imageDims, imageChannels, batchConfig, dtype, quantization, channels=[6, 12, 18, 24, 30])

class MobileNetV2Block(nn.Module):
    def __init__(self, imageDims, inputChannels, outputChannels, batchConfig, dtype, quantization, expansionFactor=6, downSample=False):
        super(MobileNetV2Block, self).__init__()

        self.downSample = downSample
        self.shortcut = (not downSample) and (inputChannels == outputChannels)
        self.imageDims = [*imageDims]

        internalChannels = inputChannels * expansionFactor

        self.inputChannels = inputChannels
        self.internalChannels = internalChannels

        self.quantizationEnabled = quantization["enabled"]

        self.conv1 = nn.Conv2d(inputChannels, internalChannels, 1, bias=False, dtype=dtype)
        self.bn1 = nn.BatchNorm2d(internalChannels, **batchConfig, dtype=dtype)
        self.conv2 = nn.Conv2d(internalChannels, internalChannels, 3, stride=2 if downSample else 1, groups=internalChannels, bias=False, padding=1, dtype=dtype)
        self.bn2 = nn.BatchNorm2d(internalChannels, **batchConfig, dtype=dtype)
        self.conv3 = nn.Conv2d(internalChannels, outputChannels, 1, bias=False, dtype=dtype)
        self.bn3 = nn.BatchNorm2d(outputChannels, **batchConfig, dtype=dtype)

    def forward(self, x):
        y = self.conv1(x)
        if self.quantizationEnabled:
            y = QNN.quantize(y)
        y = self.bn1(y)
        if self.quantizationEnabled:
            y = QNN.quantize(y)
        y = relu1(y, inplace=True)
        y = self.conv2(y)
        if self.quantizationEnabled:
            y = QNN.quantize(y)
        y = self.bn2(y)
        if self.quantizationEnabled:
            y = QNN.quantize(y)
        y = relu1(y, inplace=True)
        y = self.conv3(y)
        if self.quantizationEnabled:
            y = QNN.quantize(y)
        y = self.bn3(y)
        if self.quantizationEnabled:
            y = QNN.quantize(y)

        if self.shortcut:
            return y + x
        else:
            return y

    def getComputeCost(self):
        hiddenDims = [*self.imageDims]
        if self.downSample:
            hiddenDims[0] /= 2
            hiddenDims[1] /= 2

        return GetConvolutionalComputeCost(self.imageDims, self.inputChannels, self.internalChannels, 1) +\
            GetConvolutionalComputeCost(self.imageDims, self.internalChannels, self.internalChannels, 3, stride=2 if self.downSample else 1) +\
            GetConvolutionalComputeCost(hiddenDims, self.internalChannels, self.outputChannels, 1)

class MobileNetV2(nn.Module):
    def __init__(self, dtype, imageDims, imageChannels, batchConfig, quantization):
        super(MobileNetV2, self).__init__()

        imageDims = [*imageDims]

        self.conv0 = nn.Conv2d(imageChannels, 32, 3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(32)

        blocks = [
            MobileNetV2Block(imageDims, 32, 16, batchConfig, dtype, quantization, expansionFactor=1, downSample=False),
            MobileNetV2Block(imageDims, 16, 24, batchConfig, dtype, quantization, downSample=False),
            MobileNetV2Block(imageDims, 24, 24, batchConfig, dtype, quantization),
            MobileNetV2Block(imageDims, 24, 32, batchConfig, dtype, quantization, downSample=False),
            MobileNetV2Block(imageDims, 32, 32, batchConfig, dtype, quantization),
            MobileNetV2Block(imageDims, 32, 32, batchConfig, dtype, quantization),
            MobileNetV2Block(imageDims, 32, 64, batchConfig, dtype, quantization, downSample=True),
            MobileNetV2Block(imageDims, 64, 64, batchConfig, dtype, quantization),
            MobileNetV2Block(imageDims, 64, 64, batchConfig, dtype, quantization),
            MobileNetV2Block(imageDims, 64, 64, batchConfig, dtype, quantization),
            MobileNetV2Block(imageDims, 64, 96, batchConfig, dtype, quantization, downSample=False),
            MobileNetV2Block(imageDims, 96, 96, batchConfig, dtype, quantization),
            MobileNetV2Block(imageDims, 96, 96, batchConfig, dtype, quantization),
            MobileNetV2Block(imageDims, 96, 160, batchConfig, dtype, quantization, downSample=True),
            MobileNetV2Block(imageDims, 160, 160, batchConfig, dtype, quantization),
            MobileNetV2Block(imageDims, 160, 160, batchConfig, dtype, quantization),
            MobileNetV2Block(imageDims, 160, 320, batchConfig, dtype, quantization, downSample=False)
        ]

        for block in blocks:
            if block.downSample:
                imageDims[0] /= 2
                imageDims[1] /= 2
                block.imageDims = imageDims

        self.blocks = nn.Sequential(*blocks)

        # last conv layers and fc layer
        self.conv1 = nn.Conv2d(320, 1280, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(1280)

    def getOutputSize(self):
        return 1280

    def forward(self, x):
        y = relu1(self.bn0(self.conv0(x)))
        y = self.blocks(y)
        y = relu1(self.bn1(self.conv1(y)))
        y = adaptive_avg_pool2d(y, 1)
        y = torch.squeeze(torch.squeeze(y, -1), -1)
        return y

    def getComputeCost(self):
        computeCost = GetConvolutionalComputeCost(self.imageDims, self.imageChannels, 32, 3)

        for block in self.blockList:
            computeCost += block.getComputeCost()

        computeCost += GetConvolutionalComputeCost(self.lastLayerImageDims, 320, 1280, 1)

        return computeCost

class MobileNetV2Short(nn.Module):
    def __init__(self, imageDims, imageChannels, batchConfig, dtype, quantization):
        super(MobileNetV2Short, self).__init__()

        self.imageDims = [*imageDims]
        self.imageChannels = imageChannels

        self.conv0 = nn.Conv2d(imageChannels, 32, 3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(32)

        self.blockList = [
            MobileNetV2Block(imageDims, 32, 16, batchConfig, dtype, quantization, expansionFactor=1, downSample=False),
            MobileNetV2Block(imageDims, 16, 24, batchConfig, dtype, quantization, downSample=False),
            # MobileNetV2Block(imageDims, 24, 24, batchConfig, dtype, quantization),
            MobileNetV2Block(imageDims, 24, 32, batchConfig, dtype, quantization, downSample=False),
            # MobileNetV2Block(imageDims, 32, 32, batchConfig, dtype, quantization),
            # MobileNetV2Block(imageDims, 32, 32, batchConfig, dtype, quantization),
            MobileNetV2Block(imageDims, 32, 64, batchConfig, dtype, quantization, downSample=True),
            # MobileNetV2Block(imageDims, 64, 64, batchConfig, dtype, quantization),
            # MobileNetV2Block(imageDims, 64, 64, batchConfig, dtype, quantization),
            # MobileNetV2Block(imageDims, 64, 64, batchConfig, dtype, quantization),
            MobileNetV2Block(imageDims, 64, 96, batchConfig, dtype, quantization, downSample=False),
            # MobileNetV2Block(imageDims, 96, 96, batchConfig, dtype, quantization),
            # MobileNetV2Block(imageDims, 96, 96, batchConfig, dtype, quantization),
            MobileNetV2Block(imageDims, 96, 160, batchConfig, dtype, quantization, downSample=True),
            # MobileNetV2Block(imageDims, 160, 160, batchConfig, dtype, quantization),
            # MobileNetV2Block(imageDims, 160, 160, batchConfig, dtype, quantization),
            MobileNetV2Block(imageDims, 160, 320, batchConfig, dtype, quantization, downSample=False)
        ]

        imageDims = [*imageDims]
        for block in self.blockList:
            if block.downSample:
                imageDims[0] /= 2
                imageDims[1] /= 2
                block.imageDims = [*imageDims]

        self.blocks = nn.Sequential(*self.blockList)

        self.lastLayerImageDims = imageDims

        # last conv layers and fc layer
        self.conv1 = nn.Conv2d(320, 1280, 1, bias=False, dtype=dtype)
        self.bn1 = nn.BatchNorm2d(1280)

    def getOutputSize(self):
        return 1280

    def forward(self, x):
        y = relu1(self.bn0(self.conv0(x)))
        y = self.blocks(y)
        y = relu1(self.bn1(self.conv1(y)))
        y = adaptive_avg_pool2d(y, 1)
        y = torch.squeeze(torch.squeeze(y, -1), -1)
        return y

    def getComputeCost(self):
        computeCost = GetConvolutionalComputeCost(self.imageDims, self.imageChannels, 32, 3)

        for block in self.blockList:
            computeCost += block.getComputeCost()

        computeCost += GetConvolutionalComputeCost(self.lastLayerImageDims, 320, 1280, 1)

        return computeCost

class BYOL(nn.Module):
    def __init__(self, emaScheduler, encoderName, predictor, projector, encoder, batchNorm, dtypeName, quantization):
        super(BYOL, self).__init__()

        self.emaScheduler = emaScheduler

        dtype = getattr(torch, dtypeName)

        QNN.QuantizeTensor.nb = quantization["nb"]
        QNN.QuantizeTensor.nf = quantization["nf"]
        self.quantizationEnabled = quantization["enabled"]
        self.weightQuantizationEnabled = quantization["quantizeWeights"]

        self.onlineEncoder = globals()[encoderName](dtype=dtype, batchConfig=batchNorm, quantization=quantization, **encoder)
        self.targetEncoder = globals()[encoderName](dtype=dtype, batchConfig=batchNorm, quantization=quantization, **encoder)
        self.onlineProjector = MLP(dtype=dtype, inputSize=self.onlineEncoder.getOutputSize(), batchNorm=batchNorm, quantization=quantization, **projector)
        self.targetProjector = MLP(dtype=dtype, inputSize=self.targetEncoder.getOutputSize(), batchNorm=batchNorm, quantization=quantization, **projector)
        self.predictor = MLP(dtype=dtype, inputSize=self.onlineProjector.getOutputSize(), outputSize=self.targetProjector.getOutputSize(), batchNorm=batchNorm, quantization=quantization, **predictor)

        # Make sure the target network starts out the same as the online network
        for onlineParam, targetParam in zip(self.onlineParameters(), self.targetParameters()):
            targetParam.requires_grad = False
            targetParam.data = onlineParam.data

        if self.emaScheduler.getTau() == 0:
            print("Using SimSiam")

    def forward(self, dataView1, dataView2):
        # dimensions of dataView1,2: [batchSize, channelCount, imageWidth, imageHeight]

        if self.quantizationEnabled:
            with torch.no_grad():
                dataView1 = QNN.quantize(dataView1)
                dataView2 = QNN.quantize(dataView2)

        # Standard BYOL approach
        if self.emaScheduler.getTau() != 0:
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
        if tau != 0:
            for onlineParam, targetParam in zip(self.onlineParameters(), self.targetParameters()):
                targetParam.data = targetParam.data + (onlineParam.data - targetParam.data) * (1.0 - tau)
                targetParam.requires_grad = False

        # Simplified SimSiam approach
        else:
            for onlineParam, targetParam in zip(self.onlineParameters(), self.targetParameters()):
                targetParam.data = onlineParam.data
                targetParam.requires_grad = False

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

    def getForwardComputeCost(self):
        computeCost = 2 * (self.onlineEncoder.getComputeCost() + self.onlineProjector.getComputeCost() + self.predictor.getComputeCost() + self.targetEncoder.getComputeCost() + self.targetProjector.getComputeCost())
        computeCost[2] /= 2
        return computeCost

    def quantizeParameters(self):
        if self.quantizationEnabled and self.weightQuantizationEnabled:
            for param in self.parameters():
                with torch.no_grad():
                    param.data = QNN.quantize(param.data)
