import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

# Cargar el dataset Iris
iris = datasets.load_iris()
X = iris.data[:100, [0, 2]]  # Usaremos solo dos características para facilitar la visualización
y = iris.target[:100]

# Convertir las etiquetas a {0, 1}
y = np.where(y == 0, 0, 1)

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Estandarizar las características
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Construir el modelo de Keras
model = Sequential()
model.add(Dense(1, input_dim=2, activation='linear'))

# Compilar el modelo
model.compile(optimizer=SGD(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=1, verbose=1)

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Precisión: {accuracy:.2f}')

# Graficar la precisión
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()

# Graficar la pérdida
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# --- Visualización ---
plt.figure(figsize=(12, 6))

# Region de decisión
plt.subplot(1, 2, 1)
plot_decision_regions(X_test, y_test, ppn)
plt.title('Region de Decisión')
plt.xlabel('Longitud del sépalo')
plt.ylabel('Ancho del sépalo')
plt.legend(loc='upper left')