import numpy as np
import matplotlib.pyplot as plt
import torch

def PlotImage(image):
    plt.imshow(np.squeeze(image.detach().cpu().numpy()))
    plt.show()

def GetDeviceFromConfig(config):
    deviceName = config["device"]
    if deviceName == "cuda" and torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")