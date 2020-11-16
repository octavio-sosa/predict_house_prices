import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-1.0*x))
