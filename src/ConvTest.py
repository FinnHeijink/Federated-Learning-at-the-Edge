import Dataset
import KRIAInterface
import Config
import torch

def main():
    config = Config.GetConfig()
    dataset = Dataset.Dataset(**config["dataset"])

    _, (batch, label) = next(enumerate(dataset.testingEnumeration()))

    weight = torch.rand((1, 1, 3, 3)) #outChannels, inChannels, kernelX, kernelY
    out = KRIAInterface.FConv2D_3x3.apply(batch, weight, None)
    print(out)

if __name__ == "__main__":
    main()