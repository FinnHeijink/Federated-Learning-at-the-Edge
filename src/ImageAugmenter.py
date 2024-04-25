import torch
import torchvision.transforms as transforms

class ImageAugmenter:
    def __init__(self, imageDims):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=imageDims, antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])

    def createImagePairBatch(self, imageBatch):
        return torch.stack([self.transform(image) for image in imageBatch]), torch.stack([self.transform(image) for image in imageBatch])