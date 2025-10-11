import numpy as np 

def step( z):
    return 1 if z>0 else 0

X = np.array([[0,0],[0,1],[1,0],[1,1]])
              
y = np.array([0,1,1,1])

eta = 1 # learning rate
epochs = 3 # cantidad de épocas para el entrenamiento
w = np.array([0, 0]) # pesos iniciales
b = 0

for epoch in range(epochs):
    for i in range(len(X)):
        xi = X[i]
        target = y[i]
        z = np.dot(w, xi) + b
        y_pred = step(z)
        error = target - y_pred
        # Actualización
        w = w + eta * error * xi
        b = b + eta * error

        

print("Resultados del modelo: para OR")
for xi in X:
    y_pred = np.dot(w, xi) + b
    salida = True if y_pred > 0 else False
    print(f"Entrada {xi} -> Predicción: {salida}")

