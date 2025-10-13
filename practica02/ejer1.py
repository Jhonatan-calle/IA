import numpy as np 
import matplotlib.pyplot as plt


x = [1,2,3,4,5]
y= [2,3,5,7,8]
X = np.column_stack((np.ones_like(x),x))

Y = np.array([y])
Y = Y.T

#Ecuacion normal
coef = np.linalg.inv(X.T @ X) @ X.T @ Y

# Definir la función lineal
x_linea = np.linspace(0, 6, 100)  # Rango de x para la línea
y_linea = coef[0][0] + coef[1][0] * x_linea

# Graficar la línea de la función
plt.plot(x_linea, y_linea, color="red", label="y = 0.2 + 1.6x")

# Puntos a resaltar


# Graficar los puntos
plt.scatter(x, y, color="blue", marker="o", s=100, label="Datos")

# Opcional: resaltar cada punto con su coordenada
for xi, yi in zip(x, y):
    plt.text(xi+0.05, yi+0.05, f"({xi},{yi})")

# Etiquetas y título
plt.xlabel("Eje X")
plt.ylabel("Eje Y")
plt.title("Función lineal y puntos resaltados")
plt.legend()

# Mostrar gráfico
plt.show()
