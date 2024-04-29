import torch
from torchvision import datasets, transforms

class Dataset:

    def __init__(self, datasetName, batchSize, normalization):

        # Todo: make generic
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(normalization[0], normalization[1])])

        self.train = getattr(datasets, datasetName)('datasets', train=True, download=True, transform=transform)
        self.test = getattr(datasets, datasetName)('datasets', train=False, transform=transform)

        self.trainLoader = torch.utils.data.DataLoader(self.train, batch_size=batchSize)
        self.testLoader = torch.utils.data.DataLoader(self.test, batch_size=batchSize)

        self.batchSize = batchSize

    def trainingEnumeration(self):
        return self.trainLoader

    def testingEnumeration(self):
        return self.testLoader

    def trainBatchCount(self):
        return len(self.trainLoader.dataset)

    def testBatchCount(self):
        return len(self.testLoader.dataset)