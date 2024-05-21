import Dataset
import Config
import ImageAugmenter
import torch
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt
import numpy as np

def main():
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

if __name__ == "__main__":
    main()