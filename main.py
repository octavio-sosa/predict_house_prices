import numpy as np
import nn as nn
import data_processor as dp
import matplotlib.pyplot as plt

def main():
    X_train, X_test, Y_train, Y_test = dp.get_data()
    train_info = nn.train(X_train, Y_train, X_test, Y_test,
                       n_iter=1_000, test_every=100, learning_rate=0.001,
                       hidden_size=13, batch_size=23, 
                       return_losses=True, return_weights=True,
                       return_scores=True, seed=80718)

    losses = train_info[0]
    weights = train_info[1]
    val_scores = train_info[2]
    print(f'val_scores: {[round(s, 2) for s in val_scores]}')
    plt.xlabel('iteration')
    plt.ylabel('loss (RMSE)')
    plt.plot(losses)
    plt.show()

if __name__ == '__main__':
    main()
