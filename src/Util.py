import numpy as np
import matplotlib.pyplot as plt

def PlotImage(image):
    plt.imshow(np.squeeze(image.detach().cpu().numpy()))
    plt.show()