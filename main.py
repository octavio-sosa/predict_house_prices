import numpy as np
import nn as nn

def main():
    x = np.full((3,3), 2)
    s = nn.sigmoid(x)
    print(s)

if __name__ == '__main__':
    main()
