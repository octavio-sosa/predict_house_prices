from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def get_data():
    boston = load_boston()
    data = boston.data
    target = boston.target
    features = boston.feature_names

    scaler = StandardScaler()
    preproc_data = scaler.fit_transform(data)

    X_train, X_test, Y_train, Y_test = train_test_split(preproc_data, target,
                                                        test_size=0.3,
                                                        random_state=80718)

    reshaped_Y_train, reshaped_Y_test = Y_train.reshape(-1, 1), Y_test.reshape(-1, 1)

    return X_train, X_test, reshaped_Y_train, reshaped_Y_test
