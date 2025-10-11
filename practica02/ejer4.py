import pandas as pd

# Leer
df = pd.read_csv("advertising.csv")
print(df)

# Acceder a una columna
print(df["nombre"])

# Escribir
df.to_csv("salida.csv", index=False)
