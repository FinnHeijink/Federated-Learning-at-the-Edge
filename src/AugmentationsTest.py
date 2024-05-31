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
    nantest()