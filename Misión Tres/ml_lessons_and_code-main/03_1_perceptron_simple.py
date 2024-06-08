import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as mtr

# --- Carga del dataset iris ---
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Selecciona dos clases de flores y las dos primeras características para simplificar la visualización
X = X[y != 2][:, [0, 1]]
y = y[y != 2]

# Divide los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

class Perceptron:
    """
    Implementación del perceptrón de Rosenblatt.
    """

    def __init__(self, learning_rate=0.01, n_iterations=50):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])
        self.errors = []

        for _ in range(self.n_iterations):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

def plot_decision_regions(X, y, classifier, resolution=0.02):
    """
    Visualiza las regiones de decisión del clasificador.
    """
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=plt.cm.bwr)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx], marker=markers[idx], label=cl)

# --- Entrenamiento del perceptrón ---
ppn = Perceptron(learning_rate=0.01, n_iterations=50)
ppn.fit(X_train, y_train)

# --- Visualización ---
plt.figure(figsize=(18, 6))

# Región de decisión
plt.subplot(1, 3, 1)
plot_decision_regions(X_test, y_test, ppn)
plt.title('Región de Decisión')
plt.xlabel('Longitud del sépalo')
plt.ylabel('Ancho del sépalo')
plt.legend(loc='upper left')

# Pérdidas por época
plt.subplot(1, 3, 2)
plt.plot(range(1, len(ppn.errors) + 1), ppn.errors, marker='o')
plt.title('Pérdidas por Época')
plt.xlabel('Épocas')
plt.ylabel('Errores')

# Curva ROC
plt.subplot(1, 3, 3)
y_probs = ppn.net_input(X_test)  # Probabilidades de predicción (scores)
fpr, tpr, thresholds = mtr.roc_curve(y_test, y_probs)
roc_auc = mtr.roc_auc_score(y_test, y_probs)
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.grid(True)

plt.tight_layout()
plt.show()

# Matriz de confusión
y_pred = ppn.predict(X_test)
plt.figure(figsize=(6, 6))
conf_matrix = mtr.confusion_matrix(y_test, y_pred)
plt.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.7)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Matriz de Confusión')
plt.show()

# Evaluación
print('Accuracy: %.2f' % mtr.accuracy_score(y_test, y_pred))
print('F-1 Score: %.2f' % mtr.f1_score(y_test, y_pred))
print('AUC: %.2f' % roc_auc)
