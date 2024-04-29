import torch
import torch.optim as optim

import Dataset
import Config
import Model
import ImageAugmenter
import Communication

class DataSource:
    def startEpoch(self):
        pass

    def getData(self):
        pass

    def isEpochFinished(self):
        pass

class DatasetDataSource(DataSource):
    def __init__(self, config):
        self.dataset = Dataset.Dataset(**config["dataset"])

        self.len = len(self.dataset)
        self.index = 0

    def startEpoch(self):
        self.index = 0

    def getDataBatch(self):
        result = self.dataset[self.index]
        self.index = self.index + 1
        return result

    def isEpochFinished(self):
        return self.index == self.len

class Client:
    def __init__(self, device, config, dataSource : DataSource):
        self.device = device
        self.config = config

        self.model = Model.BYOL(**config["EMA"], **config["BYOL"]).to(device)
        self.optimizer = getattr(optim, config["optimizer"]["name"])(self.model.trainableParameters(), **config["optimizer"]["settings"])

        self.augmenter = ImageAugmenter.ImageAugmenter(**config["augmenter"])

        self.communication = Communication()

    def connect(self, port):
        self.communication.connect(port)

    def run(self):
        shouldStop = False

        epoch = 0

        while not shouldStop and epoch < self.config["training"]["epochs"]:
            self.communication.receiveModel(self.model.state_dict())

            self.dataSource.startEpoch()

            batchIndex = 0
            while not self.dataSource.isEpochFinished():
                data, target = self.getDataBatch()
                dataView1, dataView2 = self.augmenter.createImagePairBatch(data)
                dataView1, dataView2, target = dataView1.to(self.device), dataView2.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                loss = self.model(dataView1, dataView2, target)
                loss.backward()
                self.optimizer.step()
                self.model.stepEMA()

                if batchIndex % 10 == 0:
                    print(f"Epoch {epoch + 1}, batch {batchIndex}: loss={loss:.4f}")
                batchIndex = batchIndex + 1

            epoch = epoch + 1

            self.communication.sendModel(self.model.state_dict())

        self.communication.close()

def main():
    config = Config.GetConfig()

    torch.manual_seed(0)
    device = torch.device(config["device"])

    dataSource = DatasetDataSource(config)
    client = Client(device, config, dataSource)

    client.connect(1234)
    client.run()

if __name__ == "__main__":
    main()