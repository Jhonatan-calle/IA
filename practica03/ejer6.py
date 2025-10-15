import numpy as np
import matplotlib.pyplot as plt

X = np.array(
    [
        # Clase 0: Lager (baja amargura, baja oscuridad)
        [12, 15],[13, 18],[15, 20],[16, 22],[18, 25],
        [20, 27],[21, 30],[22, 31],[23, 33],[25, 35],

        # Clase 1: Stout (alta amargura, muy oscura)
        [35, 45],[38, 50],[40, 55],[42, 60],[43, 63],
        [45, 65],[46, 67],[48, 70],[50, 72],[52, 75],

        # Clase 2: IPA (amarga pero no tan oscura)
        [28, 35],[30, 38],[33, 40],[35, 42],[37, 45],
        [39, 47],[40, 50],[42, 52],[43, 54],[45, 55],

        # Clase 3: Scottish (ligeramente oscura, no tan amarga)
        [18, 32],[20, 35],[22, 38],[24, 40],[25, 43],
        [26, 45],[27, 47],[28, 48],[29, 49],[30, 50],
    ]
)

y = np.array(
    [
        0,0,0,0,0,0,0,0,0,0,      # Lager
        1,1,1,1,1,1,1,1,1,1,      # Stout
        2,2,2,2,2,2,2,2,2,2,      # IPA
        3,3,3,3,3,3,3,3,3,3,      # Scottish
    ]
)


plt.scatter(X[y==0, 0], X[y==0, 1], color="gold", label="Lager")
plt.scatter(X[y==1, 0], X[y==1, 1], color="brown", label="Stout")
plt.scatter(X[y==2, 0], X[y==2, 1], color="orange", label="IPA")
plt.scatter(X[y==3, 0], X[y==3, 1], color="purple", label="Scottish")

plt.xlabel("IBU (Amargor)")
plt.ylabel("RMS (Oscuridad)")
plt.title("Dataset sintético de estilos de cerveza")
plt.legend()
plt.show()


#TODO
