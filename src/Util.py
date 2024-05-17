import numpy as np
import matplotlib.pyplot as plt
import torch
import math

def PlotImage(image):
    plt.imshow(np.squeeze(torch.movedim(image * 0.5 + 0.5, 0, 2).detach().cpu().numpy()))
    plt.show()

def PlotStatistics(statistics):
    statistics = np.array(statistics)

    print("Statistics:", statistics)

    loss = statistics[:,0]
    accuracy = statistics[:,1]
    epochs = statistics[:,2]

    plt.plot(epochs, loss, label="Loss")
    plt.plot(epochs, accuracy, label="Accuracy")
    plt.xlabel("Epochs #")
    plt.ylim(0)
    plt.legend()
    plt.show()

def GetDeviceFromConfig(config):
    deviceName = config["device"]
    if deviceName == "cuda" and torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")

class WarmupCosineScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, startEpoch, epochCount, warmupEpochs, baseLearningRate):
        self.baseLearningRate = baseLearningRate
        self.epochCount = epochCount
        self.warmupEpochs = warmupEpochs

        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch=epochCount)

        for i in range(startEpoch):
            self.step()

    def get_lr(self):
        afterWarmupEpoch = self._step_count - self.warmupEpochs
        nonWarmupEpochCount = self.epochCount - self.warmupEpochs

        if self._step_count < self.warmupEpochs:
            return [self.baseLearningRate * self._step_count / self.warmupEpochs] # Linear warm-up
        elif self._step_count < self.epochCount:
            return [self.baseLearningRate * 0.5 * (1 + math.cos(math.pi * afterWarmupEpoch / nonWarmupEpochCount))]
        else:
            return [self.baseLearningRate]

class EMAScheduler:
    def __init__(self, initialTau, epochCount, enableSchedule):
        self.initialTau = initialTau
        self.epochCount = epochCount
        self.enableSchedule = enableSchedule

        self.tau = self.initialTau

    def step(self, currentEpoch):
        if not self.enableSchedule:
            return

        self.tau = 1 - (1 - self.initialTau) * 0.5 * (1 + math.cos(math.pi * currentEpoch / self.epochCount))

    def getTau(self):
        return self.tau