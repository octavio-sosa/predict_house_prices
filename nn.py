import numpy as np
from typing import Dict

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-1.0*x))

def init_weights(num_features: int, num_weight_combos: int)\
                -> Dict[str, np.ndarray]:
    '''
    @param num_features number of original features in data
    @param num_weight_combos aka hidden_size or num_features_to_learn
    '''
    weights: Dict[str, np.ndarray] = {}
    weights['W1'] = np.random.randn(num_features, num_weight_combos)
    weights['B1'] = np.random.randn(1, num_weight_combos)
    weights['W2'] = np.random.randn(num_weight_combos, 1)
    weights['B2'] = np.random.randn(1, 1)
    return weights


