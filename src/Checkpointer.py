import os.path as path
from enum import Enum

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

    def getModelCheckpointPath(self):
        return path.join(self.directory, "Model.pt")

    def getOptimizerCheckpointPath(self):
        return path.join(self.directory, "Optimizer.pt")

    def update(self, model, optimizer, currentEpoch, maxEpochs, currentBatch, maxBatches):
        pass

    def loadLastCheckpoint(self, model, optimizer):
        pass

    def saveCheckpoint(self, model, optimizer):
        torch.save(model.state_dict(), self.getModelCheckpointPath())

        if self.saveOptimizerData:
            torch.save(optimizer.state_dict(), self.getOptimizerCheckpointPath())