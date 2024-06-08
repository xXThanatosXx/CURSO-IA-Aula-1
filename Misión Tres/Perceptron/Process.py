from sklearn import datasets
from sklearn.model_selection import train_test_split

def load_and_prepare_data():
    iris = datasets.load_()
    X = iris.data
    y = iris.target

    X = X[y != 2][:, [0, 1]]
    y = y[y != 2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test
