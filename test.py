import numpy as np
import nn as nn

def test():
    data = np.full((3,3), 2)
    s = nn.sigmoid(data)
    weights = nn.init_weights(data.shape[1], data.shape[1])
    print(weights)

test()
