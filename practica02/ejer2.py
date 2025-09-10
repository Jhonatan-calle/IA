import numpy as np
import matplotlib.pyplot as plt

# Fijar semilla para reproducibilidad
np.random.seed(42)

# Generar valores de x
x = np.linspace(0, 10, 10)  # 10 puntos entre 0 y 10

# Generar ruido e ~ N(0,1)
e = np.random.normal(0, 1, size=x.shape)

# Calcular y
y = 3 * x + 2 + e
Y = y.reshape(-1, 1) #cantidad de filas que hagan falta y solo 1 columna 

# ----- Solución por ecuación normal -----
X = np.column_stack((np.ones_like(x), x))  # matriz (m,2)
coef_normal = np.linalg.inv(X.T @ X) @ X.T @ Y  # (2,1)
coef_normal = coef_normal.flatten() 


# ----- Preparar para Gradiente Descendente -----
m = X.shape[0] #guarda el nro de filas
X_b = X.copy()            # (m, 2) con columna de 1s ya incluida

#inicializa una matriz con ceros los cuales van a ser los coificientes iniciales 
coef_aux_gr = np.zeros((2, 1))  
eta = 0.01                # tasa de aprendizaje (tamaño de los saltos)
n_iter = 1000             # iteracionesN


# Gradiente descendente vectorizado
for iteration in range(n_iter):
    gradients = (1 / m) * X_b.T @ (X_b @ coef_aux_gr - Y)  # (2,1) estudiar gradiente y calculo de errro medio
    coef_aux_gr = coef_aux_gr - eta * gradients

coef_aux_gr = coef_aux_gr.flatten()

# ----- Resultados y comparación -----
print("Coeficientes (ecuación normal):")
print(f"  theta0 = {coef_normal[0]:.6f}, theta1 = {coef_normal[1]:.6f}")
print("Coeficientes (gradiente descendente):")
print(f"  theta0 = {coef_aux_gr[0]:.6f}, theta1 = {coef_aux_gr[1]:.6f}")

# Calcular rectas para graficar
y_line_normal = coef_normal[0] + coef_normal[1] * x
y_line_grad = coef_aux_gr[0] + coef_aux_gr[1] * x
y_true = 3 * x + 2

# ----- Gráficos -----

# Subplot 1: datos y rectas
plt.scatter(x, y, label="Datos sintéticos", marker='o')
plt.plot(x, y_true, linestyle='--', label="y = 3x + 2 (sin ruido)")
plt.plot(x, y_line_normal, color='red', label=f"Ecuación normal: y = {coef_normal[1]:.3f}x + {coef_normal[0]:.3f}")
plt.plot(x, y_line_grad, color='green', linestyle='-.', label=f"Gradiente: y = {coef_aux_gr[1]:.3f}x + {coef_aux_gr[0]:.3f}")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Ajuste lineal")
plt.legend()


plt.show()
