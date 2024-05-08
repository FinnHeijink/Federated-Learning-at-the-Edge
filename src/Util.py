import numpy as np
import matplotlib.pyplot as plt
import torch

def PlotImage(image):
    plt.imshow(np.squeeze(image.detach().cpu().numpy()))
    plt.show()

def PlotStatistics(statistics):
    statistics = np.array(statistics)
    loss = statistics[:,0]
    accuracy = statistics[:,1]

    epochs = np.arange(0, len(loss))

    plt.plot(epochs, loss, label="Loss")
    plt.plot(epochs, accuracy, label="Accuracy")
    plt.xlabel("Epochs #")
    plt.ylim(0)
    plt.legend()
    plt.show()

def GetDeviceFromConfig(config):
    deviceName = config["device"]
    if deviceName == "cuda" and torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")