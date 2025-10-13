import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1 Generar dataset sintético
np.random.seed(42)
X = np.linspace(0, 10, 20).reshape(-1, 1)  # 20 puntos entre 0 y 10
y = X**2 + np.random.normal(0, 5, size=X.shape)  # y = x^2 + ruido


X_poli = np.column_stack((np.ones_like(X),X**2))

# 2 Ajustar regresión lineal
lin_reg = LinearRegression() #inicialicaliacion con coeficionetes nulos 
lin_reg.fit(X_poli, y) # Calcula la mejor pendiente y el intercepto que minimizan el error cuadrático medio.
y_pred_poly = lin_reg.predict(X_poli)

# 3 Calcular MSE
mse_poly = mean_squared_error(y, y_pred_poly)
print(f"MSE de la regresión polinomial: {mse_poly:.2f}")
print(f"Coeficientes: pendiente = {lin_reg.coef_[0][0]:.2f}, intercepto = {lin_reg.coef_[0,1]:.2f}")

# 4 Graficar
plt.scatter(X, y, color='blue', label="Datos sintéticos")
plt.plot(X, y_pred_poly, color='red', label="Regresión polinomial")
plt.plot(X, X**2, color='green', linestyle='--', label="Función original y=x^2")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Ajuste de regresión lineal a datos cuadráticos")
plt.legend()
plt.grid(True)
plt.show()
