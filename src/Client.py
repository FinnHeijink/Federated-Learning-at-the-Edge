import torch
import torch.optim as optim

import time

import Dataset
import Config
import Model
import ImageAugmenter
import Communication
import Util

class DataSource:

    def initBuffer(self, model, device):
        pass

    def updateBuffer(self, model, device):
        pass

    def startEpoch(self):
        pass

    def getDataBatch(self): # Should not return target labels
        pass

    def isEpochFinished(self):
        pass

class DataBufferDataSource(DataSource):

    class DataBufferImage:
        def __init__(self, image, score, timeout):
            self.image = image
            self.score = score
            self.timeout = timeout

    def __init__(self, config):

        config = config.copy()
        config["dataset"]["batchSize"] = config["dataBuffer"]["datasetLoadBatchSize"] #Don't load complete batches from the dataset for filling the image buffer

        self.config = config

        self.dataset = Dataset.Dataset(**config["dataset"])
        self.enumeration = self.dataset.trainingEnumeration()

        self.augmenter = ImageAugmenter.ImageAugmenter(**config["augmenter"])

        self.buffer = [] # array of (image, score)
        self.bufferTargetSize = config["dataBuffer"]["bufferSize"]
        self.lazyScoringInterval = config["dataBuffer"]["lazyScoringInterval"]
        self.batchSize = config["dataBuffer"]["batchSize"]
        self.epochStreamCount = config["dataBuffer"]["epochStreamCount"]

        self.index = 0

    def calculateScore(self, model, device, image):
        image = torch.unsqueeze(image, 0)
        dataView1, dataView2 = self.augmenter.weaklyAugment(image)
        dataView1, dataView2 = dataView1.to(device), dataView2.to(device)

        model.eval()
        with torch.no_grad():
            loss = model(dataView1, dataView2)

        return loss.item()

    def updateAndStreamNewData(self, model, device, newImageCount):

        # Update the score of the images in the buffer when necessary (lazy scoring)
        rescoreCount = 0
        for bufferImage in self.buffer:
            bufferImage.timeout = bufferImage.timeout - 1
            if bufferImage.timeout == 0:
                bufferImage.score = self.calculateScore(model, device, bufferImage.image)
                bufferImage.timeout = self.lazyScoringInterval
                rescoreCount = rescoreCount + 1
        print(f"Rescored {rescoreCount} images")

        # Load newImageCount new images in batches, and add them to the buffer, keeping the images in the buffer with the highest scores
        imagesProcessed = 0
        newImagesAcceptedCount = 0
        while imagesProcessed < newImageCount:
            batchData, batchLabels = next(iter(self.enumeration))
            for image in batchData:
                bufferImage = self.DataBufferImage(image, self.calculateScore(model, device, image), self.lazyScoringInterval)
                imagesProcessed = imagesProcessed + 1

                self.buffer.append(bufferImage)

                if len(self.buffer) > self.bufferTargetSize:

                    # Buffer is now too large, remove the item with the lowest score

                    lowestIndex = -1
                    lowestScore = 100000000
                    for index, bufferImage in enumerate(self.buffer):
                        if bufferImage.score < lowestScore:
                            lowestScore = bufferImage.score
                            lowestIndex = index

                    if lowestIndex != len(self.buffer) - 1:
                        newImagesAcceptedCount += 1

                    self.buffer.pop(lowestIndex)
                else:
                    newImagesAcceptedCount += 1
        print(f"Accepted {newImagesAcceptedCount} new images")

    def printBuffer(self):
        for index, bufferImage in enumerate(self.buffer):
            print(f"Image #{index}: score={bufferImage.score}, timeout={bufferImage.timeout}")

    def initBuffer(self, model, device):
        self.updateAndStreamNewData(model, device, self.bufferTargetSize)

    def updateBuffer(self, model, device):
        self.updateAndStreamNewData(model, device, self.epochStreamCount)

    def startEpoch(self):
        self.index = 0

    def getDataBatch(self):
        dataAmountToFetch = min(self.batchSize, len(self.buffer) - self.index)
        images = torch.stack([bufferImage.image for bufferImage in self.buffer[-self.index - dataAmountToFetch-1:-self.index-1]])
        self.index += dataAmountToFetch
        return images

    def getProgress(self):
        return self.index / len(self.buffer)

    def isEpochFinished(self):
        return self.index == len(self.buffer)

class DatasetDataSource(DataSource):
    def __init__(self, config):
        self.dataset = Dataset.Dataset(**config["dataset"])
        self.enumeration = self.dataset.trainingEnumeration()

        self.len = self.dataset.trainBatchCount() // self.dataset.batchSize
        self.index = 0

    def startEpoch(self):
        self.index = 0

    def getDataBatch(self):
        data, target = next(iter(self.enumeration))
        self.index = self.index + 1
        return data # Note: we don't return the target labels!

    def getProgress(self):
        return self.index / self.len

    def isEpochFinished(self):
        return self.index == self.len

class Client:
    def __init__(self, device, config, dataSource : DataSource):
        self.device = device
        self.config = config
        self.dataSource = dataSource

        self.emaScheduler = Util.EMAScheduler(**config["EMA"]) # Todo: update EMA
        self.model = Model.BYOL(self.emaScheduler, **config["BYOL"]).to(device)
        self.optimizer = getattr(optim, config["optimizer"]["name"])(self.model.trainableParameters(), **config["optimizer"]["settings"])

        self.augmenter = ImageAugmenter.ImageAugmenter(**config["augmenter"])

        self.communication = Communication.Communication()

    def connect(self, ip, port):
        print(f"Connecting to {ip}:{port}")
        self.communication.connect(ip, port)

    def run(self):
        try:
            self.run_()
        except KeyboardInterrupt:
            self.communication.sendMessage("stop")
            while not self.communication.isDataReady():
                time.sleep(1)
            self.communication.receiveMessage()
            self.communication.close()

    def run_(self):
        shouldStop = False

        epoch = 0

        self.dataSource.initBuffer(self.model, self.device)

        while not shouldStop and epoch < self.config["training"]["epochs"]:

            if epoch % self.config["client"]["serverSyncEveryNEpochs"] == 0:
                print("Loading model from server")
                self.communication.sendMessage("requestSend")
                self.communication.receiveModel(self.model)

            print(f"Training BYOL Epoch {epoch + 1}: lr={self.optimizer.param_groups[0]['lr']}")
            self.dataSource.startEpoch()

            self.model.train()

            batchIndex = 0
            while not self.dataSource.isEpochFinished():
                data = self.dataSource.getDataBatch()
                dataView1, dataView2 = self.augmenter.createImagePairBatch(data)
                dataView1, dataView2 = dataView1.to(self.device), dataView2.to(self.device)

                self.optimizer.zero_grad()
                loss = self.model(dataView1, dataView2)
                loss.backward()
                self.optimizer.step()
                self.model.stepEMA()

                if batchIndex % 1 == 0:
                    print(f"Epoch {epoch + 1}, batch {batchIndex}/{self.dataSource.getProgress() * 100:.1f}%: loss={loss:.4f}")
                batchIndex = batchIndex + 1

            if epoch % self.config["client"]["updateBufferEveryNEpochs"] == 0:
                print("Updating buffer...")
                self.dataSource.updateBuffer(self.model, self.device)

            epoch = epoch + 1

            # Note: checking after epoch+1! We load from server at the first epoch in a sequence, and send to the server at the last of the sequence
            if epoch % self.config["client"]["serverSyncEveryNEpochs"] == 0:
                print("Sending model to server")
                self.communication.sendMessage("update")
                self.communication.sendModel(self.model)

        self.communication.close()

def main():
    config = Config.GetConfig()

    #torch.manual_seed(0)
    device = Util.GetDeviceFromConfig(config)

    Model.SetUseReLU1(config["useReLU1"])

    #dataSource = DatasetDataSource(config)
    dataSource = DataBufferDataSource(config)
    client = Client(device, config, dataSource)

    client.connect("localhost", 1234)
    client.run()

if __name__ == "__main__":
    main()