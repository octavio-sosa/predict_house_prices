import numpy as np
from typing import Dict, Tuple

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

def forward_loss(observations: np.ndarray, targets: np.ndarray, weights: Dict[str, np.ndarray])\
                -> Tuple[Dict[str, np.ndarray], float]:
    M1 = np.dot(observations, weights['W1'])
    N1 = M1 + weights['B1']
    O1 = sigmoid(N1)
    M2 = np.dot(O1, weights['W2'])
    predictions = M2 + weights['B2']
    loss = np.mean(np.power(targets - predictions, 2))

    forward_info: Dict[str, np.ndarray] = {}
    forward_info['X'] = observations
    forward_info['M1'] = M1
    forward_info['N1'] = N1
    forward_info['O1'] = O1
    forward_info['M2'] = M2
    forward_info['P'] = predictions
    forward_info['Y'] = targets

    return forward_info, loss

