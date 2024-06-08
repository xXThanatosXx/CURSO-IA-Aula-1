import matplotlib.pyplot as plt
import sklearn.metrics as mtr
from Process import load_and_prepare_data
from Perceptron import Perceptron
import grapics

def main():
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    ppn = Perceptron(learning_rate=0.0001, n_iterations=100)  # Ajuste de hiperparámetros
    ppn.fit(X_train, y_train)

    plt.figure(figsize=(18, 6))

    # Región de decisión
    plt.subplot(1, 3, 1)
    grapics.plot_decision_regions(X_test, y_test, ppn)
    plt.title('Región de Decisión')
    plt.xlabel('Feature 2')
    plt.ylabel('Feature 8')
    plt.legend(loc='upper left')

    # Pérdidas por época
    plt.subplot(1, 3, 2)
    grapics.plot_errors(ppn.errors)

    # Curva ROC
    plt.subplot(1, 3, 3)
    y_probs = ppn.net_input(X_test)
    grapics.plot_roc_curve(y_test, y_probs)

    plt.tight_layout()
    plt.show()

    # Matriz de confusión
    y_pred = ppn.predict(X_test)
    plt.figure(figsize=(6, 6))
    grapics.plot_confusion_matrix(y_test, y_pred)
    plt.show()

    # Evaluación
    print('Accuracy: %.2f' % mtr.accuracy_score(y_test, y_pred))
    print('F-1 Score: %.2f' % mtr.f1_score(y_test, y_pred))
    roc_auc = mtr.roc_auc_score(y_test, y_probs)
    print('AUC: %.2f' % roc_auc)

if __name__ == '__main__':
    main()