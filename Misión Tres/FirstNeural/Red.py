import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, roc_auc_score, confusion_matrix

# Datos
celcius = np.array([-40,-10,0,8,15,22,38], dtype=float)
fahrenheit = np.array([-40,14,32,46,59,72,100], dtype=float)

# Definición del modelo
# capa = tf.keras.layers.Dense(units=1, input_shape=[1])
# modelo = tf.keras.Sequential([capa])

oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1,oculta2,salida])
# Compilación del modelo
modelo.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')

# Entrenamiento del modelo
print("Comenzando entrenamiento")
historial = modelo.fit(celcius, fahrenheit, epochs=1000, verbose=False)
print("Modelo entrenado")

# Gráfico de la pérdida
plt.figure(figsize=(8, 6))
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.show()

#Test de prediccion
resultado = modelo.predict([100.0])
print("El resultado es " + str(resultado)+ "fahrenheit")

#Variables
print("Variables internas del modelo")
# print(capa.get_weights())
print(modelo.get_weights())

modelo.save('.\Misión Tres\FirstNeural\celsius.h5')

# # Convertir el problema a uno de clasificación binaria
# # Por ejemplo, clasificar por encima y por debajo de 32 grados Fahrenheit
# labels_reales = (fahrenheit >= 32).astype(int)
# labels_predichos = (predicciones >= 32).astype(int)

# # Cálculo de métricas
# precision = precision_score(labels_reales, labels_predichos)
# auc = roc_auc_score(labels_reales, predicciones)

# print(f"Precisión: {precision}")
# print(f"AUC: {auc}")

# # Matriz de confusión
# matriz_confusion = confusion_matrix(labels_reales, labels_predichos)
# print("Matriz de confusión:")
# print(matriz_confusion)


