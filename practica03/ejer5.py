import numpy as np
import matplotlib.pyplot as plt


X = np.array(
    [
        # Clase 0 (por ejemplo: Lager)
        [12, 15],
        [13, 18],
        [15, 20],
        [16, 22],
        [18, 25],
        [20, 27],
        [21, 30],
        [22, 31],
        [23, 33],
        [25, 35],
        # Clase 1 (por ejemplo: Stout)
        [35, 45],
        [38, 50],
        [40, 55],
        [42, 60],
        [43, 63],
        [45, 65],
        [46, 67],
        [48, 70],
        [50, 72],
        [52, 75],
    ]
)

y = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,  # Lager
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,  # Stout
    ]
)


# Funcion sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Modelo
def modelo(X, Y, learning_rate, iterations):
    X = X.T  # trasponemos para realizar la multiplicación de matrices
    n = X.shape[0]  # cantidad de características
    m = X.shape[1]  # cantidad de casos
    W = np.zeros((n, 1))  # vector de pesos para cada característica
    B = 0

    for i in range(iterations):
        Z = np.dot(W.T, X) + B  # mult pesos por casos
        A = sigmoid(Z)  # tenemos el vector de resultados de cada caso
        # Función de costo
        costo = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        # Aplicación de la técnica Gradient Descent
        dW = (1 / m) * np.dot(A - Y, X.T)
        dB = (1 / m) * np.sum(A - Y)
        # Ajuste de pesos
        W = W - learning_rate * dW.T
        B = B - learning_rate * dB
        if i % (iterations / 10) == 0:
            print("costo luego de iteración", i, "es : ", costo)
    return W, B


# entrenamiento
W, B = modelo(X, y, learning_rate=0.001, iterations=100000)

print("\nPesos finales:\n", W)
print("Bias final:", B)

# Predicciones
Z = np.dot(W.T, X.T) + B
A = sigmoid(Z)
pred = (A > 0.5).astype(int)

print("\nPredicciones:", pred.flatten())
print("Reales:      ", y.flatten())

# Graficar resultados
# plt.figure(figsize=(6, 4))
# for i, label in enumerate(y.flatten()):
#     color = "red" if label == 1 else "blue"
#     plt.scatter(X[i, 0], X[i, 1], color=color)
#
# # Frontera de decisión (línea donde probabilidad = 0.5)
# x_vals = np.linspace(10, 50, 100)
# y_vals = -(W[0] * x_vals + B) / W[1]
# plt.plot(x_vals, y_vals.flatten(), "--", color="green")
#
# plt.xlabel("IBU")
# plt.ylabel("RMS")
# plt.title("Regresión logística - Clasificación Lager vs Stout")
# plt.show()


# validacion
X_val = np.array(
    [
        [14, 19],
        [19, 26],
        [24, 34],
        [37, 48],
        [41, 58],
        [47, 68],
    ]
)

y_val = np.array([0, 0, 0, 1, 1, 1])

Z = np.dot(W.T, X_val.T) + B
A = sigmoid(Z)
pred = (A > 0.5).astype(int).flatten()

TP = 0
FP = 0
FN = 0
TN = 0

for i in range(len(y_val)):
    if pred[i]== 1:
        if y_val[i]==1:
            TP+=1
        else:
            FP+=1
    else:
        if y_val[i]==1:
            FN+=1
        else:
            TN+=1
    
# Exactitud (accuracy)
accuracy = (TP + TN) / len(y_val)
# Precisión (reliability)
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
# Exhaustividad (recall)
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
# F1-Score (calidad general)
f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) != 0 else 0

print("\nPrecisión (Precision):", precision)
print("Exactitud (Accuracy):", accuracy)
print("Exhaustividad (Recall):", recall)
print("Calidad general (F1):", f1)
