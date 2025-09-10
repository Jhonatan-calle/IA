import numpy as np

# Datos de ejemplo: y = 4 + 3x + ruido
np.random.seed(42)
m = 100
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)

# Preparamos X con columna de 1s (bias)
X_b = np.c_[np.ones((m, 1)), X]

# Inicializamos parámetros
theta = np.zeros((2, 1))  # θ0 y θ1
eta = 0.1                 # tasa de aprendizaje
n_iter = 10000000

# Función de costo (MSE)
def mse(X_b, y, theta):
    return (1/m) * np.sum((X_b @ theta - y) ** 2)

# Entrenamiento
for iteration in range(n_iter):
    gradients = (2/m) * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients
    
    if iteration % 100 == 0:  # cada 100 iteraciones mostramos costo
        print(f"Iter {iteration}: Costo = {mse(X_b, y, theta):.4f}")

print("Theta final:", theta.ravel())
