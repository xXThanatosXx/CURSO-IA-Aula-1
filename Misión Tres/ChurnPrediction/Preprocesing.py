from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
# Normalización de los datos
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
print(X)
# Estandarización de los datos
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X)

