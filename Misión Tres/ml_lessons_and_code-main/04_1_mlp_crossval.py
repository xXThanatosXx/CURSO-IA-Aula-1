import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_predict

# Cargar el conjunto de datos Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Nombres de las columnas (características)
feature_names = iris.feature_names

# Mostrar los nombres de las columnas
print("Nombres de las columnas (características):")
for name in feature_names:
    print(name)

# Escalar las características para mejorar el rendimiento del MLP
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Crear un MLP
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=10000, alpha=0.001,
                    solver='adam', tol=0.0001, random_state=1,
                    learning_rate_init=.01)

# Usar cross_val_predict para obtener predicciones usando validación cruzada
y_pred = cross_val_predict(mlp, X_scaled, y, cv=5)

# Matriz de confusión
cm = confusion_matrix(y, y_pred)

# Imprimir matriz de confusión
print("\nMatriz de Confusión:")
print(cm)

# Visualizar matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="YlGnBu", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Imprimir métricas de rendimiento
print("\nReporte de Clasificación:")
print(classification_report(y, y_pred, target_names=iris.target_names))

print("Exactitud (Accuracy):", accuracy_score(y, y_pred))
