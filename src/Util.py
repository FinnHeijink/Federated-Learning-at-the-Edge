import numpy as np
import matplotlib.pyplot as plt
import torch

def PlotImage(image):
    plt.imshow(np.squeeze(image.detach().cpu().numpy()))
    plt.show()

def GetDeviceFromConfig(config):
    deviceName = config["device"]
    if deviceName == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")