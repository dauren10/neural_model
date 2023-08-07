import numpy as np

def sigmoid(x, der=False):
    if der == True:
        return x * (1-x)
    else:
        1 / (1 + np.exp(-x))


