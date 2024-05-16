import os.path as path
import time
import datetime
import glob
import os
from enum import Enum

import torch

class CheckpointMode(Enum):
    DISABLED = -1,
    EVERY_EPOCH = 0,
    EVERY_BATCH = 1,
    EVERY_N_BATCHES = 2,
    EVERY_N_SECS = 3,

class Checkpointer:

    def __init__(self, directory, checkpointMode, saveOptimizerData=False, checkPointEveryNSecs=0, checkPointEveryNBatches=0, prefix=""):
        self.directory = directory
        self.mode = checkpointMode
        self.saveOptimizerData = saveOptimizerData
        self.checkPointEveryNSecs = checkPointEveryNSecs
        self.checkPointEveryNBatches = checkPointEveryNBatches

        self.lastEpoch = -1
        self.lastBatch = -1
        self.lastCheckpointTime = -1

        self.prefix = prefix
        self.runIdentifier = datetime.datetime.now().strftime("%d%m%y_%H%M%S")

    def getModelCheckpointPath(self):
        return path.join(self.directory, self.prefix + "Model_" + self.runIdentifier + "_" + str(self.lastEpoch) + ".pt")

    def getOptimizerCheckpointPath(self):
        return path.join(self.directory, self.prefix + "Optimizer_" + self.runIdentifier + "_" + str(self.lastEpoch) + ".pt")

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

        self.lastEpoch = currentEpoch

    def loadLastCheckpoint(self, model, optimizer):
        listOfFiles = glob.glob(self.directory + "/*")
        latestFile = max(listOfFiles, key=lambda f: os.path.getctime(f) if f.split("\\")[-1].startswith(self.prefix) else 0).split("\\")[-1]

        if not latestFile.startswith(self.prefix):
            print(f"No {self.prefix} checkpoint found")
            return

        postfix = "_".join(latestFile.split("_")[1:])

        self.loadCheckpointFromPostfix(postfix, model, optimizer)

    def loadCheckpointFromPostfix(self, postfix, model, optimizer):
        print(f"Loading {self.prefix} checkpoint: {postfix}")

        modelPath = path.join(self.directory, self.prefix + "Model_" + postfix)
        if os.path.exists(modelPath):
            model.load_state_dict(torch.load(modelPath))

        if optimizer and self.saveOptimizerData:
            optimizerPath = path.join(self.directory, self.prefix + "Optimizer_" + postfix)
            if os.path.exists(optimizerPath):
                optimizer.load_state_dict(torch.load(optimizerPath))

    def loadCheckpoint(self, specificCheckpoint, model, optimizer):
        if specificCheckpoint == None:
            self.loadLastCheckpoint(model, optimizer)
        else:
            self.loadCheckpointFromPostfix(specificCheckpoint, model, optimizer)

    def saveCheckpoint(self, model, optimizer):
        torch.save(model.state_dict(), self.getModelCheckpointPath())

        if self.saveOptimizerData:
            torch.save(optimizer.state_dict(), self.getOptimizerCheckpointPath())