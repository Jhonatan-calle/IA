import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# 1️⃣ Generar dataset sintético
np.random.seed(42)
X_training = np.linspace(0, 10, 20).reshape(-1, 1)
y_training = (1/3) * X_training**4  +  1/8 + np.random.normal(0, 100, size=X_training.shape)

X_validation = np.linspace(0.25, 10.25, 20).reshape(-1, 1)  
y_validation = (1/3) * X_validation**4  + 1/8


# 2️⃣ Transformar dataset para regresión polinomial
grado = 3   # puedes cambiar a 3, 4, etc. para observar transformaciones
poly = PolynomialFeatures(degree=grado)
X_poly = poly.fit_transform(X_training)

# Imprimir dataset transformado
print("Dataset transformado (primeras 5 filas):")
print(X_poly[:5])

# 3️⃣ Ajustar modelo polinomial
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y_training)
y_pred_poly = lin_reg_poly.predict(X_poly)

# 4️⃣ Ajustar modelo lineal simple para comparación
lin_reg_lineal = LinearRegression()
lin_reg_lineal.fit(X_training, y_training)
y_pred_lineal = lin_reg_lineal.predict(X_training)

# 5️⃣ Calcular MSE
mse_poly = mean_squared_error(y_training, y_pred_poly)
print(f"\nMSE modelo polinomial (grado {grado}): {mse_poly:.2f}")

# 6️⃣ Graficar
plt.scatter(X_validation, y_validation, color='green', label="Datos validacion")
plt.scatter(X_training, y_training, color='blue', label="Datos entrenamiento")
plt.plot(X_training, y_pred_poly, color='red', label=f"Polinomial grado {grado}")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparación de difententes grados de fit")
plt.legend()
plt.grid(True)
plt.show()
