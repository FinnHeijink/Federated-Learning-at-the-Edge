import os.path as path
from enum import Enum
import time

import torch

class CheckpointMode(Enum):
    DISABLED = -1,
    EVERY_EPOCH = 0,
    EVERY_BATCH = 1,
    EVERY_N_BATCHES = 2,
    EVERY_N_SECS = 3,

class Checkpointer:

    def __init__(self, directory, checkpointMode, saveOptimizerData=False, checkPointEveryNSecs=0, checkPointEveryNBatches=0):
        self.directory = directory
        self.checkpointMode = checkpointMode
        self.saveOptimizerData = saveOptimizerData
        self.checkPointEveryNSecs = checkPointEveryNSecs
        self.checkPointEveryNBatches = checkPointEveryNBatches

        self.lastEpoch = -1
        self.lastBatch = -1
        self.lastCheckpointTime = -1

    def getModelCheckpointPath(self):
        return path.join(self.directory, "Model.pt")

    def getOptimizerCheckpointPath(self):
        return path.join(self.directory, "Optimizer.pt")

    def update(self, model, optimizer, currentEpoch, maxEpochs, currentBatch, maxBatches):
        currentTime = time.time()

        if self.mode == CheckpointMode.EVERY_EPOCH:
            if currentEpoch != self.lastEpoch:
                self.lastEpoch = currentEpoch
                self.saveCheckpoint(model, optimizer)
        elif self.mode == CheckpointMode.EVERY_BATCH:
            if currentEpoch != self.lastEpoch or currentBatch != self.lastBatch:
                self.lastEpoch = currentEpoch
                self.lastBatch = currentBatch
                self.saveCheckpoint(model, optimizer)
        elif self.mode == CheckpointMode.EVERY_N_SECS:
            if currentTime > self.lastCheckpointTime + self.checkPointEveryNSecs:
                self.lastCheckpointTime = currentTime
                self.saveCheckpoint(model, optimizer)
        elif self.mode == CheckpointMode.EVERY_N_BATCHES:
            raise NotImplementedError
        elif self.mode == CheckpointMode.DISABLED:
            pass
        else:
            raise NotImplementedError

    def loadLastCheckpoint(self, model, optimizer):
        pass

    def saveCheckpoint(self, model, optimizer):
        torch.save(model.state_dict(), self.getModelCheckpointPath())

        if self.saveOptimizerData:
            torch.save(optimizer.state_dict(), self.getOptimizerCheckpointPath())