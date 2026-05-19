import pandas as pd

# Ruta absoluta dinámica (funciona desde cualquier lugar)
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
df = pd.read_csv(os.path.join(BASE_DIR, "data", "telco_churn.csv"))

# ---- Cargar Dataset ----
#df = pd.read_csv("data/telco_churn.csv")

# -- vemos el tipo de dato de cada columna --
# -- vemos si hay valores nulos --
#print(df.info())

# -- vemos columnas categoricas --
#print(df.select_dtypes(include=['object']).columns)

# -- vemos columnas mas detalladamente --
columnas_categoricas = df.select_dtypes(include=['object']).columns.tolist()
print(f"Columnas categóricas ({len(columnas_categoricas)}):")
for col in columnas_categoricas:
    print(f"  • {col}")