import numpy as np
import matplotlib.pyplot as plt

# --- Datos sintéticos ---
np.random.seed(42)
x = np.linspace(0, 10, 10)
e = np.random.normal(0, 1, size=x.shape)
y = 3 * x + 2 + e
Y = y.reshape(-1, 1)  # (m,1)

# --- Ecuación normal ---
X = np.column_stack((np.ones_like(x), x))  # (m,2)
coef_normal = np.linalg.inv(X.T @ X) @ X.T @ Y
coef_normal = coef_normal.flatten()

# --- Gradiente descendente ---
m = X.shape[0]
X_b = X.copy()
coef_aux_gr = np.zeros((2, 1))
eta = 0.01
n_iter = 1000

def mse(X_b, coef_aux_gr, Y):
    error = X_b @ coef_aux_gr - Y
    return (error).mean()

convergence_history = []

for iteration in range(n_iter):
    gradients = (1 / m) * X_b.T @ (X_b @ coef_aux_gr - Y)
    coef_aux_gr = coef_aux_gr - eta * gradients
    convergence_history.append(mse(X_b, coef_aux_gr, Y))

coef_aux_gr = coef_aux_gr.flatten()

# --- Resultados ---
print("Coeficientes (ecuación normal):")
print(f"  theta0 = {coef_normal[0]:.6f}, theta1 = {coef_normal[1]:.6f}")
print("Coeficientes (gradiente descendente):")
print(f"  theta0 = {coef_aux_gr[0]:.6f}, theta1 = {coef_aux_gr[1]:.6f}")

y_line_normal = coef_normal[0] + coef_normal[1] * x
y_line_grad = coef_aux_gr[0] + coef_aux_gr[1] * x
y_true = 3 * x + 2

# --- Gráficos combinados ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Subplot 1: datos y rectas
ax1.scatter(x, y, label="Datos sintéticos", marker='o')
ax1.plot(x, y_true, linestyle='--', label="y = 3x + 2 (sin ruido)")
ax1.plot(x, y_line_normal, color='red',
         label=f"Ecuación normal: y = {coef_normal[1]:.3f}x + {coef_normal[0]:.3f}")
ax1.plot(x, y_line_grad, color='green', linestyle='-.',
         label=f"Gradiente: y = {coef_aux_gr[1]:.3f}x + {coef_aux_gr[0]:.3f}")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("Ajuste lineal")
ax1.legend()
ax1.grid(True)

# Subplot 2: convergencia del error
ax2.plot(convergence_history, color='purple')
ax2.set_xlabel("Iteraciones")
ax2.set_ylabel("MSE")
ax2.set_title("Convergencia del gradiente descendente")
ax2.grid(True)

plt.tight_layout()
plt.show()
