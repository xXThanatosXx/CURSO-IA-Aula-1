<h1 align="center">Métricas</h1>

### Comparación Matriz de Confusión
```python
y_true = [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]
y_pred = [0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0]

# Calcular Verdaderos Negativos (TN)
TN = sum((y_true[i] == 1) and (y_pred[i] == 0) for i in range(len(y_true)))

# Calcular Falsos Positivos (FP)
FP = sum((y_true[i] == 0) and (y_pred[i] == 1) for i in range(len(y_true)))

# Calcular Falsos Negativos (FN)
FN = sum((y_true[i] == 0) and (y_pred[i] == 0) for i in range(len(y_true)))

# Calcular Verdaderos Positivos (TP)
TP = sum((y_true[i] == 1) and (y_pred[i] == 1) for i in range(len(y_true)))

print(f"Verdaderos Negativos (TN): {TN}")
print(f"Falsos Positivos (FP): {FP}")
print(f"Falsos Negativos (FN): {FN}")
print(f"Verdaderos Positivos (TP): {TP}")
```

## Ejemplo Matriz de Confusión

Supongamos que estamos trabajando en un problema de clasificación binaria donde estamos tratando de predecir si un correo electrónico es spam (1) o no spam (0). Hemos entrenado nuestro modelo y ahora queremos evaluarlo usando una matriz de confusión.

Paso 1: Recopilar Predicciones y Etiquetas Reales
Primero, recopilamos las predicciones de nuestro modelo y las etiquetas reales de un conjunto de prueba.

### Paso 1: Recopilar datos
```python
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Supongamos que tenemos las siguientes etiquetas reales y predicciones del modelo
y_true = [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]
y_pred = [0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0]

# Crear la matriz de confusión
cm = confusion_matrix(y_true, y_pred)

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Spam', 'Spam'], yticklabels=['No Spam', 'Spam'])
plt.xlabel('Clase Predicha')
plt.ylabel('Clase Real')
plt.title('Matriz de Confusión')
plt.show()


```

### Paso 2: Interpretar la Matriz de Confusión
```python
# Generar un informe de clasificación
report = classification_report(y_true, y_pred, target_names=['No Spam', 'Spam'])
print(report)
```

<h1 align="center">sklearn.metrics</h1>

La biblioteca `sklearn.metrics` de scikit-learn proporciona una amplia gama de funciones para evaluar la precisión y el rendimiento de los modelos de aprendizaje automático. A continuación se detallan algunas de las funcionalidades principales que permite realizar:

### Métricas de clasificación

- **Exactitud (Accuracy):** `accuracy_score()`
- **Precisión, Recall y F1-score:** `precision_score()`, `recall_score()`, `f1_score()`
- **Curvas ROC y AUC:** `roc_curve()`, `auc()`
- **Curvas PR (Precision-Recall):** `precision_recall_curve()`, `average_precision_score()`
- **Matriz de confusión:** `confusion_matrix()`
- **Informe de clasificación:** `classification_report()`
- **Puntaje de log-loss:** `log_loss()`

### Métricas de regresión

- **Error cuadrático medio (MSE):** `mean_squared_error()`
- **Error absoluto medio (MAE):** `mean_absolute_error()`
- **Coeficiente de determinación (R^2):** `r2_score()`
- **Error absoluto mediano:** `median_absolute_error()`
- **Error cuadrático logarítmico medio:** `mean_squared_log_error()`

### Métricas de clustering

- **Índice de Rand ajustado:** `adjusted_rand_score()`
- **Coeficiente de silueta:** `silhouette_score()`
- **Homogeneidad, completitud y V-measure:** `homogeneity_score()`, `completeness_score()`, `v_measure_score()`
- **Índice de Davies-Bouldin:** `davies_bouldin_score()`
- **Índice de Calinski-Harabasz:** `calinski_harabasz_score()`

### Métricas de pares

- **Coeficiente de correlación de Pearson:** `pearsonr()`
- **Coeficiente de correlación de Spearman:** `spearmanr()`
- **Coeficiente de correlación de Kendall Tau:** `kendalltau()`

### Otros

- **Puntaje ROC AUC:** `roc_auc_score()`
- **Puntaje de clasificación cruzada:** `cross_val_score()`
- **Coeficiente de correlación de Matthews (MCC):** `matthews_corrcoef()`
- **Logaritmo de la pérdida de probabilidad:** `log_loss()`

Estas funciones permiten evaluar la efectividad de los modelos y seleccionar el mejor modelo para el conjunto de datos específico. Además, muchas de estas métricas se pueden personalizar y ajustar según las necesidades del proyecto.
