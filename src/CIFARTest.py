import torch
import torch.nn as nn
from torch.nn.functional import relu, max_pool2d, log_softmax, nll_loss, relu6, adaptive_avg_pool2d
import torch.optim as optim
from torchvision import datasets, transforms

batchSize = 64
learningRate = 0.001
schedulerGamma = 0.85
epochs = 10

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = self.dropout1(max_pool2d(x, 2))
        x = relu(self.fc1(torch.flatten(x, 1)))
        x = self.dropout2(x)
        y = log_softmax(self.fc2(x), dim=1)
        return y

class MobileNetV2Block(nn.Module):
    def __init__(self, inputChannels, outputChannels, expansionFactor=6, downSample=False):
        super(MobileNetV2Block, self).__init__()

        self.downSample = downSample
        self.shortcut = (not downSample) and (inputChannels == outputChannels)

        internalChannels = inputChannels * expansionFactor

        self.conv1 = nn.Conv2d(inputChannels, internalChannels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(internalChannels)
        self.conv2 = nn.Conv2d(internalChannels, internalChannels, 3, stride=2 if downSample else 1, groups=internalChannels, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(internalChannels)
        self.conv3 = nn.Conv2d(internalChannels, outputChannels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(outputChannels)

    def forward(self, x):
        y = relu6(self.bn1(self.conv1(x)))
        y = relu6(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))

        if self.shortcut:
            return y + x
        else:
            return y

class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()

        self.conv0 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(32)

        self.blocks = nn.Sequential(
            MobileNetV2Block(32, 16, expansionFactor=1, downSample=False),
            MobileNetV2Block(16, 24, downSample=False),
            #MobileNetV2Block(24, 24),
            MobileNetV2Block(24, 32, downSample=False),
            #MobileNetV2Block(32, 32),
            #MobileNetV2Block(32, 32),
            MobileNetV2Block(32, 64, downSample=True),
            #MobileNetV2Block(64, 64),
            #MobileNetV2Block(64, 64),
            #MobileNetV2Block(64, 64),
            MobileNetV2Block(64, 96, downSample=False),
            #MobileNetV2Block(96, 96),
            #MobileNetV2Block(96, 96),
            MobileNetV2Block(96, 160, downSample=True),
            #MobileNetV2Block(160, 160),
            #MobileNetV2Block(160, 160),
            MobileNetV2Block(160, 320, downSample=False))

        # last conv layers and fc layer
        self.conv1 = nn.Conv2d(320, 1280, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(1280)

        self.fc = nn.Linear(1280, 10)

    def forward(self, x):
        y = relu6(self.bn0(self.conv0(x)))
        y = self.blocks(y)
        y = relu6(self.bn1(self.conv1(y)))
        y = adaptive_avg_pool2d(y, 1)
        y = torch.squeeze(torch.squeeze(y, -1), -1)
        y = self.fc(y)
        y = log_softmax(y, dim=1)
        return y

def train(model, device, trainLoader, optimizer, epoch):
    model.train() #Enables dropout

    print(f"Epoch {epoch}: lr={optimizer.param_groups[0]['lr']}")

    for batchIndex, (data, target) in enumerate(trainLoader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batchIndex % 10 == 0:
            print(f"Epoch {epoch}, batch {batchIndex}/{(batchIndex*batchSize)/len(trainLoader.dataset)*100:.1f}%: loss={loss:.2f}")

def test(model, device, testLoader):
    model.eval() #Disables dropout

    testLoss = 0
    accuracy = 0

    with torch.no_grad():
        for data, target in testLoader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            testLoss += nll_loss(output, target, reduction='sum').item()
            prediction = output.argmax(dim=1, keepdim=True)
            accuracy += prediction.eq(target.view_as(prediction)).sum().item()

    testLoss /= len(testLoader.dataset)
    accuracy /= len(testLoader.dataset)

    print(f"Evaluation: loss={testLoss:2f}, accuracy={accuracy*100:.1f}%")

def main():
    torch.manual_seed(0)
    device = torch.device("cuda")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    dsTrain = datasets.CIFAR10('datasets', train=True, download=True, transform=transform)
    dsTest = datasets.CIFAR10('datasets', train=False, transform=transform)

    trainLoader = torch.utils.data.DataLoader(dsTrain, batch_size=batchSize)
    testLoader = torch.utils.data.DataLoader(dsTest, batch_size=batchSize)

    #model = Net().to(device)
    model = MobileNetV2().to(device)
    opt = optim.Adam(model.parameters(), lr=learningRate)

    scheduler = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=schedulerGamma)
    for epoch in range(1, epochs+1):
        train(model, device, trainLoader, opt, epoch)
        test(model, device, testLoader)
        scheduler.step()

        torch.save(model.state_dict(), f"src/checkpoints/cifar_cnn_{epoch}.pt")

if __name__ == '__main__':
    main()