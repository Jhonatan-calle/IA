import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# 1️⃣ Generar dataset sintético
np.random.seed(42)
X = np.linspace(0, 10, 20).reshape(-1, 1)  # 20 puntos entre 0 y 10
y = X**2   # y = x^2 + ruido

# 2️⃣ Transformar dataset para regresión polinomial
grado = 2  # puedes cambiar a 3, 4, etc. para observar transformaciones
poly = PolynomialFeatures(degree=grado)
X_poly = poly.fit_transform(X)

# Imprimir dataset transformado
print("Dataset transformado (primeras 5 filas):")
print(X_poly[:5])

# 3️⃣ Ajustar modelo polinomial
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)
y_pred_poly = lin_reg_poly.predict(X_poly)

# 4️⃣ Ajustar modelo lineal simple para comparación
lin_reg_lineal = LinearRegression()
lin_reg_lineal.fit(X, y)
y_pred_lineal = lin_reg_lineal.predict(X)

# 5️⃣ Calcular MSE
mse_poly = mean_squared_error(y, y_pred_poly)
mse_lineal = mean_squared_error(y, y_pred_lineal)
print(f"\nMSE modelo polinomial (grado {grado}): {mse_poly:.2f}")
print(f"MSE modelo lineal: {mse_lineal:.2f}")

# 6️⃣ Graficar
plt.scatter(X, y, color='blue', label="Datos sintéticos")
plt.plot(X, y_pred_poly, color='red', label=f"Polinomial grado {grado}")
plt.plot(X, y_pred_lineal, color='orange', linestyle='--', label="Lineal")
plt.plot(X, X**2, color='green', linestyle=':', label="Función original y=x^2")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparación de regresión lineal y polinomial")
plt.legend()
plt.grid(True)
plt.show()
