from enum import Enum

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

    def update(self, model, optimizer, currentEpoch, maxEpochs, currentBatch, maxBatches):
        pass

    def loadLastCheckpoint(self, model, optimizer):
        pass

    def saveCheckpoint(self, model, optimizer):
        pass