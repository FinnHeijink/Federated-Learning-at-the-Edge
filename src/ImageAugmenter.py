import torch
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F

class ToHalfTensor(torch.nn.Module):
    def forward(self, img):
        return img.half()

class UnityTransform(torch.nn.Module):
    def forward(self, img):
        return img

class ImageAugmenter:
    def __init__(self, imageDims, applyFlips=False, applyColorAugments=False, useHalfPrecision=False):
        self.transform = transforms.Compose([
            transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur((3, 3), (1.0, 2.0))]), p=0.2),
            transforms.RandomRotation(degrees=30),
            transforms.RandomResizedCrop(size=imageDims, antialias=True, scale=(0.5, 1)),
        ])

        if applyColorAugments:
            self.transform = transforms.Compose([
                self.transform,
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
                transforms.RandomGrayscale(p=0.2),
                #transforms.GaussianBlur(9, sigma=(0.1, 0.2)),
                transforms.RandomSolarize(threshold=0.5, p=0.2)
            ])

        if applyFlips:
            self.transform = transforms.Compose([
                self.transform,
                transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomVerticalFlip(p=0.5),
            ])


        if useHalfPrecision:
            self.transform = transforms.Compose([
                self.transform,
                ToHalfTensor()
            ])
            self.weakTransform = transforms.Compose([
                transforms.RandomResizedCrop(size=imageDims, antialias=True, scale=(0.8, 1)),
                ToHalfTensor()
            ])
            self.noTransform = ToHalfTensor()
        else:
            self.weakTransform = transforms.RandomResizedCrop(size=imageDims, antialias=True, scale=(0.8, 1))
            self.noTransform = UnityTransform()

    def createImagePairBatch(self, imageBatch):
        return torch.stack([self.transform(image) for image in imageBatch]), torch.stack([self.transform(image) for image in imageBatch])

    def createImagePairBatchSingleAugment(self, imageBatch):
        return self.noTransform(imageBatch), torch.stack([self.transform(image) for image in imageBatch])

    def weaklyAugment(self, image):
        return self.noTransform(image), self.weakTransform(image)