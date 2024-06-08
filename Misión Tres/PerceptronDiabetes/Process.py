from sklearn import datasets
from sklearn.model_selection import train_test_split

def load_and_prepare_data():
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target

    # Convertir el problema en clasificación binaria:
    y = (y > y.mean()).astype(int)

    # Seleccionar dos características para simplificar la visualización
    X = X[:, [2, 8]]  # Ajuste de características seleccionadas para mejor visualización

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test