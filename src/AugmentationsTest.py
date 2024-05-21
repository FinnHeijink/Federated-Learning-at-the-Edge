import Dataset
import Config
import torch
import torchvision.transforms.v2 as transforms

def main():
    config = Config.GetConfig()
    dataset = Dataset.Dataset(**config["dataset"])

    _, batch = next(enumerate(dataset.testingEnumeration()))

    transformList = [
        transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur((3, 3), (1.0, 2.0))]), p=0.2),
        transforms.RandomRotation(degrees=30),
        transforms.RandomResizedCrop(size=config["augmenter"]["imageDims"], antialias=True, scale=(0.5, 1)),
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(9, sigma=(0.1, 0.2)),
        transforms.RandomSolarize(threshold=0.5, p=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
    ]

    N = 1000

    for i in range(N):
        x = transformList[0](batch)

    for i in range(N):
        x = transformList[1](batch)

    for i in range(N):
        x = transformList[2](batch)

    for i in range(N):
        x = transformList[3](batch)

    for i in range(N):
        x = transformList[4](batch)

    for i in range(N):
        x = transformList[5](batch)

    for i in range(N):
        x = transformList[6](batch)

    for i in range(N):
        x = transformList[7](batch)

    print(x)

if __name__ == "__main__":
    main()