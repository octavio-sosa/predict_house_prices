import numpy as np
from typing import Dict, Tuple

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-1.0*x))

def permute_data(X: np.ndarray, Y: np.ndarray):
    perm = np.random.permutation(X.shape[0])
    return X[perm], Y[perm]

def generate_batch(X: np.ndarray, Y: np.ndarray,
                   start: int = 0,
                   batch_size: int = 10):
    assert X.ndim == Y.ndim == 2

    # resize last batch_size
    if start+batch_size > X.shape[0]:
        batch_size = X.shape[0]-start

    X_batch, Y_batch = X[start:start+batch_size], Y[start:start+batch_size]
    return X_batch, Y_batch

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

def loss_gradients(forward_info: Dict[str, np.ndarray],
                   weights: Dict[str, np.ndarray])\
                  -> Dict[str, np.ndarray]:
    '''
    Compute partial derivatives of loss w.r.t. nn params
    '''

    dLdP = -2*(forward_info['Y'] - forward_info['P']) #1
    dPdM2 = np.ones_like(forward_info['M2'])#2
    dLdM2 = dLdP*dPdM2
    dM2dO1 = weights['W2'].T #3
    dO1dN1 = sigmoid(forward_info['N1'])*(1-sigmoid(forward_info['N1'])) #4
    dN1dM1 = np.ones_like(forward_info['M1']) #5
    dM1dW1 = forward_info['X'].T #6
    dLdN1 = dLdM2.dot(dM2dO1)*dO1dN1 #1,2,3,4
    dN1dB1 = np.ones_like(weights['B1'])
    dM2dW2 = forward_info['O1'].T
    dPdB2 = np.ones_like(weights['B2'])

    dLdW1 = dM1dW1.dot(dLdN1*dN1dM1) 
    dLdB1 = (dLdN1*dN1dB1).sum(axis=0)
    dLdW2 = dM2dW2.dot(dLdM2)
    dLdB2 = (dLdP*dPdB2).sum(axis=0)


    loss_gradients: Dict[str, np.ndarray] = {}
    loss_gradients['W1'] = dLdW1
    loss_gradients['B1'] = dLdB1 
    loss_gradients['W2'] = dLdW2
    loss_gradients['B2'] = dLdB2

    return loss_gradients

def predict(X: np.ndarray, weights: Dict[str, np.ndarray]) -> np.ndarray:
    M1 = X.dot(weights['W1'])
    N1 = M1+weights['B1']
    O1 = sigmoid(N1)
    M2 = O1.dot(weights['W2'])
    P = M2+weights['B2']

    return P



