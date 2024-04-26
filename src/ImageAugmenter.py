import torch
import torchvision.transforms as transforms

class ImageAugmenter:
    def __init__(self, imageDims):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=imageDims, antialias=True, scale=(0.6, 1)),
            transforms.RandomAdjustSharpness(0, p=0.5),
            transforms.RandomRotation(degrees=20)
            #transforms.RandomApply(transforms.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3),
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomVerticalFlip(p=0.5),
        ])

    def createImagePairBatch(self, imageBatch):
        return torch.stack([self.transform(image) for image in imageBatch]), torch.stack([self.transform(image) for image in imageBatch])