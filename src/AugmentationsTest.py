import Dataset
import Config
import ImageAugmenter
import Util

import torch
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt
import numpy as np

def visualize():
    config = Config.GetConfig()
    dataset = Dataset.Dataset(**config["dataset"])
    augmenter = ImageAugmenter.ImageAugmenter(**config["augmenter"])

    _, (batch, target) = next(enumerate(dataset.testingEnumeration()))
    augmentedView1, augmentedView2 = augmenter.createImagePairBatch(batch)

    plt.figure(figsize=(15, 10))
    for i, (original, view1, view2) in enumerate(zip(batch, augmentedView1, augmentedView2)):
        if i == 8:
            break

        plt.subplot(3, 8, i + 1 + 8 * 0)
        plt.imshow(np.squeeze(torch.movedim(original * 0.5 + 0.5, 0, 2).detach().cpu().numpy()))
        plt.subplot(3, 8, i + 1 + 8 * 1)
        plt.imshow(np.squeeze(torch.movedim(view1 * 0.5 + 0.5, 0, 2).detach().cpu().numpy()))
        plt.subplot(3, 8, i + 1 + 8 * 2)
        plt.imshow(np.squeeze(torch.movedim(view2 * 0.5 + 0.5, 0, 2).detach().cpu().numpy()))

    plt.tight_layout()
    plt.show()

def visualize_different_augmentations():
    config = Config.GetConfig()
    dataset = Dataset.Dataset(**config["dataset"])
    augmenter = ImageAugmenter.ImageAugmenter(**config["augmenter"])

    _, (batch, target) = next(enumerate(dataset.testingEnumeration()))

    image = batch[0]
    aug1 = transforms.GaussianBlur((3, 3), (1.0, 2.0))(image)
    aug2 = transforms.RandomRotation(degrees=30)(image)
    aug3 = transforms.RandomResizedCrop(size=(28,28), antialias=True, scale=(0.5, 1))(image)

    plt.figure(figsize=(10, 3))

    plt.subplot(1, 4, 1)
    plt.imshow(np.squeeze(torch.movedim(image * 0.5 + 0.5, 0, 2).detach().cpu().numpy()), cmap='gray')
    plt.title("Input")
    plt.subplot(1, 4, 2)
    plt.imshow(np.squeeze(torch.movedim(aug1 * 0.5 + 0.5, 0, 2).detach().cpu().numpy()), cmap='gray')
    plt.title("Blur")
    plt.subplot(1, 4, 3)
    plt.imshow(np.squeeze(torch.movedim(aug2 * 0.5 + 0.5, 0, 2).detach().cpu().numpy()), cmap='gray')
    plt.title("Rotation")
    plt.subplot(1, 4, 4)
    plt.imshow(np.squeeze(torch.movedim(aug3 * 0.5 + 0.5, 0, 2).detach().cpu().numpy()), cmap='gray')
    plt.title("Resized-crop")

    plt.tight_layout()
    plt.savefig("Augmentations.svg")
    plt.show()

def schedule_visualizer():
    config = Config.GetConfig()

    network = torch.nn.Linear(42, 69)
    optimizer = torch.optim.SGD(network.parameters())
    lrScheduler = Util.WarmupCosineScheduler(optimizer, 0, config["training"]["epochs"],
                                         config["training"]["warmupEpochs"], config["optimizer"]["settings"]["lr"])

    emaScheduler = Util.EMAScheduler(**config["EMA"])

    epochs = np.arange(config["training"]["epochs"])
    lrs = []
    taus = []
    for i in epochs:
        lrs.append(optimizer.param_groups[0]["lr"])
        taus.append(emaScheduler.getTau())

        lrScheduler.step()
        emaScheduler.step(i)

    plt.plot(epochs, lrs)
    plt.xlabel("Epoch #")
    plt.ylabel("Learning Rate")
    plt.savefig("LearningRateSchedule.svg")
    plt.show()

    plt.plot(epochs, taus)
    plt.xlabel("Epoch #")
    plt.ylabel("$\\tau$")
    plt.savefig("TauSchedule.svg")
    plt.show()


def nantest():
    torch.manual_seed(1)
    config = Config.GetConfig()
    dataset = Dataset.Dataset(**config["dataset"])
    augmenter = ImageAugmenter.ImageAugmenter(**config["augmenter"])
    device = Util.GetDeviceFromConfig(config)

    maxTrainBatches = dataset.trainBatchCount() / dataset.batchSize

    for epoch in range(2):
        for batchIndex, (data, target) in enumerate(dataset.trainingEnumeration()):
            #data = data.to(device)
            dataView1, dataView2 = augmenter.createImagePairBatch(data)

            if torch.isnan(dataView1).any() or torch.isnan(dataView2).any():
                print("Nan detected!")
                exit(1)

            if batchIndex % 100 == 0:
                print(f"At batch {batchIndex}/{batchIndex / maxTrainBatches * 100:.1f}% of epoch {epoch + 1}")

if __name__ == "__main__":
    schedule_visualizer()