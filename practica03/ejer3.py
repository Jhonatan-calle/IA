import numpy as np


X = np.array([[0, 0], [0, 1], [1,0],[2,3],[2, 2], [3, 2]])

y = np.array([0, 0, 0, 1,1, 1])

X_test = np.array([[1,1],[3,3]])

y_test = np.array([0, 1])

iter = 1000
learningRate = 0.1
weights = np.zeros(3)  # dos pesos más el bias


def activacion(z):
    return 1 if z > 0 else 0


for _ in range(iter):
    errores = 0
    for xi, target in zip(X, y):
        z = np.dot(xi, weights[1:]) + weights[0]
        y_pred = activacion(z)
        error = target - y_pred
        weights[1:] += learningRate * error * xi
        weights[0] += learningRate * error
        errores += abs(error)
    if errores == 0:
        break

print("Pesos finales:", weights)

print("\nResultados de test:")
for xi, target in zip(X_test, y_test):
    z = np.dot(xi, weights[1:]) + weights[0]
    y_pred = activacion(z)
    print(f"Entrada: {xi}, Esperado: {target}, Predicho: {y_pred}")


#si agrego el (3,3,0) los dotos dejan de ser linealmente separables por lo que nunca se van a encontrar 
#pesos que funcionen correctamente para todos los datos
