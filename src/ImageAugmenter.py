import torch
import torchvision.transforms as transforms

class ImageAugmenter:
    def __init__(self, imageDims, applyFlips=False, applyColorAugments=False):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=imageDims, antialias=True, scale=(0.5, 1)),
            transforms.RandomAdjustSharpness(0, p=0.5),
            transforms.RandomRotation(degrees=30)
        ])

        if applyColorAugments:
            self.transform = transforms.Compose([
                self.transform,
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.4),
                transforms.RandomGrayscale(p=0.3)
            ])

        if applyFlips:
            self.transform = transforms.Compose([
                self.transform,
                transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomVerticalFlip(p=0.5),
            ])

        self.weakTransform = transforms.RandomResizedCrop(size=imageDims, antialias=True, scale=(0.8, 1))

    def createImagePairBatch(self, imageBatch):
        return torch.stack([self.transform(image) for image in imageBatch]), torch.stack([self.transform(image) for image in imageBatch])

    def createImagePairBatchSingleAugment(self, imageBatch):
        return imageBatch, torch.stack([self.transform(image) for image in imageBatch])

    def weaklyAugment(self, image):
        return image, self.weakTransform(image)