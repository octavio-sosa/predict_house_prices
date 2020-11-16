import numpy as np
import nn as nn
import data_processor as dp


def weights():
    data = np.full((3,3), 2)
    s = nn.sigmoid(data)
    weights = nn.init_weights(data.shape[1], data.shape[1])
    print(weights)

def data():
    X_train, X_test, Y_train, Y_test = dp.get_data()
    data = {'X_train': X_train, 'X_test': X_test, 'Y_train': Y_train, 'Y_test': Y_test}
    for key in data.keys():
        print(f'{key}: {data[key].shape}')

def forward():
    X_train, X_test, Y_train, Y_test = dp.get_data()
    

def test():
    data()
    #forward()

test()
