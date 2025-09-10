import numpy as np 
import matplotlib.pyplot as plt


X = np.array([
            [1,1],
            [1,2],
            [1,3],
            [1,4],
            [1,5]
            ])

Y = np.array([[2,3,5,7,8]])
Y = Y.T

aux1 = X.T @ X
print(aux1)
coef = np.linalg.inv(aux1) @ X.T @ Y

# Definir la función lineal
x_linea = np.linspace(0, 6, 100)  # Rango de x para la línea
y_linea = coef[0][0] + coef[1][0] * x_linea

# Graficar la línea de la función
plt.plot(x_linea, y_linea, color="red", label="y = 0.2 + 1.6x")

# Puntos a resaltar
puntos = { (1,2), (2,3), (3,5), (4,7), (5,8) }
x_puntos = [p[0] for p in puntos]
y_puntos = [p[1] for p in puntos]

# Graficar los puntos
plt.scatter(x_puntos, y_puntos, color="blue", marker="o", s=100, label="Datos")

# Opcional: resaltar cada punto con su coordenada
for xi, yi in zip(x_puntos, y_puntos):
    plt.text(xi+0.05, yi+0.05, f"({xi},{yi})")

# Etiquetas y título
plt.xlabel("Eje X")
plt.ylabel("Eje Y")
plt.title("Función lineal y puntos resaltados")
plt.legend()

# Mostrar gráfico
plt.show()
