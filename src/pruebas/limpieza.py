import pandas as pd

# ---- Cargar Dataset ----
df = pd.read_csv("data/telco_churn.csv")

# -- vemos el tipo de dato de cada columna --
# -- vemos si hay valores nulos --
print(df.info())

# -- vemos columnas categoricas --
df.select_dtypes(include=['object']).columns

