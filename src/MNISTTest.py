import torch
import torch.nn as nn
from torch.nn.functional import relu, max_pool2d, log_softmax, nll_loss
import torch.optim as optim
from torchvision import datasets, transforms

batchSize = 64
learningRate = 0.001
schedulerGamma = 0.7
epochs = 10

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = self.dropout1(max_pool2d(x, 2))
        x = relu(self.fc1(torch.flatten(x, 1)))
        x = self.dropout2(x)
        y = log_softmax(self.fc2(x), dim=1)
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

    batchCount = 0

    with torch.no_grad():
        for data, target in testLoader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            testLoss += nll_loss(output, target, reduction='sum').item()
            prediction = output.argmax(dim=1, keepdim=True)
            accuracy += prediction.eq(target.view_as(prediction)).sum().item()/len(data)
            batchCount += 1

    testLoss /= batchCount
    accuracy /= batchCount

    print(f"Evaluation: loss={testLoss:2f}, accuracy={accuracy*100:.1f}%")

def main():
    torch.manual_seed(0)
    device = torch.device("cuda")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    dsTrain = datasets.MNIST('datasets', train=True, download=True, transform=transform)
    dsTrain, _ = torch.utils.data.random_split(dsTrain, [0.1, 0.9])
    dsTest = datasets.MNIST('datasets', train=False, transform=transform)
    #dsTrain.data.to(device)
    #dsTrain.targets.to(device)

    trainLoader = torch.utils.data.DataLoader(dsTrain, batch_size=batchSize)
    testLoader = torch.utils.data.DataLoader(dsTest, batch_size=batchSize)

    model = Net().to(device)
    opt = optim.Adam(model.parameters(), lr=learningRate)

    scheduler = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=schedulerGamma)
    for epoch in range(1, epochs+1):
        train(model, device, trainLoader, opt, epoch)
        test(model, device, testLoader)
        scheduler.step()

        torch.save(model.state_dict(), f"src/checkpoints/mnist_cnn_{epoch}.pt")

if __name__ == '__main__':
    main()