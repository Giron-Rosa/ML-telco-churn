# =============================================================
#  src/preprocessing/clean.py
#  Proyecto : Predicción de Abandono de Clientes (Churn)
#  Etapa    : 2 — Limpieza y Codificación de Variables
#  Ejecutar : python src/preprocessing/clean.py
# =============================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', palette='Set2')

# ── Rutas del proyecto ────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'figuras')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Columnas por tipo de encoding ─────────────────────────
#    (definidas aquí para tenerlas visibles y documentadas)

# Binarias con solo 2 valores → Label Encoding (0 / 1)
COLS_LABEL = [
    'gender',           # Male / Female
    'Partner',          # Yes / No
    'Dependents',       # Yes / No
    'PhoneService',     # Yes / No
    'MultipleLines',    # Yes / No / No phone service
    'OnlineSecurity',   # Yes / No / No internet service
    'OnlineBackup',     # Yes / No / No internet service
    'DeviceProtection', # Yes / No / No internet service
    'TechSupport',      # Yes / No / No internet service
    'StreamingTV',      # Yes / No / No internet service
    'StreamingMovies',  # Yes / No / No internet service
    'PaperlessBilling', # Yes / No
    'Contract',         # Month-to-month / One year / Two year (ordinal)
]

# Nominales con más de 2 categorías → One-Hot Encoding
COLS_OHE = [
    'InternetService',  # DSL / Fiber optic / No
    'PaymentMethod',    # 4 categorías sin orden
]

# Variable objetivo → mapeo manual Yes=1 / No=0
COL_TARGET = 'Churn'

# Columna a eliminar
COL_DROP = 'customerID'


# =============================================================
#  FUNCIONES
# =============================================================

def cargar_dataset(nombre='telco_churn.csv'):
    """Carga el dataset original desde data/."""
    ruta = os.path.join(DATA_DIR, nombre)
    df = pd.read_csv(ruta)
    print('=' * 60)
    print('  LIMPIEZA Y CODIFICACIÓN DEL DATASET')
    print('=' * 60)
    print(f'  Dataset cargado: {nombre}')
    print(f'  Filas: {df.shape[0]:,}  |  Columnas: {df.shape[1]}')
    return df


# ─────────────────────────────────────────────────────────────
#  PASO 1 — Eliminar columnas innecesarias
# ─────────────────────────────────────────────────────────────

def eliminar_columnas(df):
    """Elimina customerID ya que no aporta información predictiva."""
    print('\n── PASO 1: Eliminar columnas innecesarias ──────────────')
    df = df.drop(columns=[COL_DROP])
    print(f'  ✅ Columna eliminada: {COL_DROP}')
    print(f'  Columnas restantes: {df.shape[1]}')
    return df


# ─────────────────────────────────────────────────────────────
#  PASO 2 — Corregir tipos de datos
# ─────────────────────────────────────────────────────────────

def corregir_tipos(df):
    """
    TotalCharges viene como 'object' por espacios vacíos.
    Se convierte a float y los espacios pasan a NaN.
    """
    print('\n── PASO 2: Corregir tipos de datos ─────────────────────')

    tipo_antes = df['TotalCharges'].dtype
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    tipo_despues = df['TotalCharges'].dtype

    nulos_generados = df['TotalCharges'].isnull().sum()

    print(f'  TotalCharges: {tipo_antes} → {tipo_despues}')
    print(f'  Valores NaN generados: {nulos_generados}')
    return df


# ─────────────────────────────────────────────────────────────
#  PASO 3 — Tratar valores nulos
# ─────────────────────────────────────────────────────────────

def tratar_nulos(df):
    """
    Imputa los NaN de TotalCharges con la mediana agrupada
    por tipo de contrato. Es más preciso que la mediana global
    porque cada tipo de contrato tiene rangos de gasto distintos.
    """
    print('\n── PASO 3: Tratar valores nulos ────────────────────────')

    nulos_antes = df['TotalCharges'].isnull().sum()
    print(f'  Nulos antes : {nulos_antes}')

    # Imputación por mediana agrupada por Contract
    mediana_por_contrato = df.groupby('Contract')['TotalCharges'].transform('median')
    df['TotalCharges'] = df['TotalCharges'].fillna(mediana_por_contrato)

    nulos_despues = df['TotalCharges'].isnull().sum()
    print(f'  Nulos después: {nulos_despues}')
    print(f'  ✅ Imputación con mediana agrupada por Contract')

    # Verificación general de nulos
    total_nulos = df.isnull().sum().sum()
    print(f'  Total nulos en dataset: {total_nulos}')
    return df


# ─────────────────────────────────────────────────────────────
#  PASO 4 — Codificar variable objetivo
# ─────────────────────────────────────────────────────────────

def codificar_target(df):
    """Convierte Churn: Yes → 1, No → 0."""
    print('\n── PASO 4: Codificar variable objetivo (Churn) ─────────')

    df[COL_TARGET] = df[COL_TARGET].map({'Yes': 1, 'No': 0})

    distribucion = df[COL_TARGET].value_counts()
    pct_churn    = df[COL_TARGET].mean() * 100

    print(f'  ✅ Churn codificado: Yes=1 / No=0')
    print(f'  Distribución:')
    print(f'    0 (No Churn) : {distribucion[0]:,}')
    print(f'    1 (Churn)    : {distribucion[1]:,}  ({pct_churn:.1f}%)')
    return df


# ─────────────────────────────────────────────────────────────
#  PASO 5 — Label Encoding
# ─────────────────────────────────────────────────────────────

def aplicar_label_encoding(df):
    """
    Aplica Label Encoding a columnas binarias y ordinales.

    Mapeos especiales:
    - gender          : Male=1, Female=0
    - Contract        : Month-to-month=0, One year=1, Two year=2
    - Columnas Yes/No : Yes=1, No=0
    - Columnas con 'No internet/phone service': tratadas como No=0
    """
    print('\n── PASO 5: Label Encoding ──────────────────────────────')

    # Mapeo para columnas con Yes / No / No internet service / No phone service
    mapeo_yes_no = {
        'Yes'                 : 1,
        'No'                  : 0,
        'No internet service' : 0,   # equivalente a "No" para el modelo
        'No phone service'    : 0,
    }

    # Mapeo específico por columna
    mapeos_especiales = {
        'gender'  : {'Male': 1, 'Female': 0},
        'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    }

    for col in COLS_LABEL:
        if col in mapeos_especiales:
            df[col] = df[col].map(mapeos_especiales[col])
            print(f'  ✅ {col:<20} → {mapeos_especiales[col]}')
        else:
            df[col] = df[col].map(mapeo_yes_no)
            print(f'  ✅ {col:<20} → Yes=1 / No=0')

    return df


# ─────────────────────────────────────────────────────────────
#  PASO 6 — One-Hot Encoding
# ─────────────────────────────────────────────────────────────

def aplicar_ohe(df):
    """
    Aplica One-Hot Encoding a InternetService y PaymentMethod.
    drop_first=True elimina una columna por variable para evitar
    multicolinealidad (dummy variable trap).
    """
    print('\n── PASO 6: One-Hot Encoding ────────────────────────────')

    cols_antes = df.shape[1]

    for col in COLS_OHE:
        print(f'  Valores únicos en {col}: {df[col].unique().tolist()}')

    df = pd.get_dummies(df, columns=COLS_OHE, drop_first=True, dtype=int)

    cols_despues   = df.shape[1]
    cols_agregadas = cols_despues - cols_antes

    print(f'\n  ✅ OHE aplicado a: {COLS_OHE}')
    print(f'  Columnas antes  : {cols_antes}')
    print(f'  Columnas después: {cols_despues}  (+{cols_agregadas} nuevas)')

    # Mostrar las columnas nuevas generadas
    nuevas_cols = [c for c in df.columns if any(
        c.startswith(base + '_') for base in COLS_OHE
    )]
    print(f'  Columnas generadas: {nuevas_cols}')

    return df


# ─────────────────────────────────────────────────────────────
#  PASO 7 — Verificación final
# ─────────────────────────────────────────────────────────────

def verificar_resultado(df):
    """Verifica que no queden columnas tipo object ni nulos."""
    print('\n── PASO 7: Verificación final ──────────────────────────')

    # Columnas object restantes
    cols_object = df.select_dtypes(include=['object']).columns.tolist()
    if cols_object:
        print(f'  ⚠️  Columnas object restantes: {cols_object}')
    else:
        print('  ✅ No quedan columnas de tipo object')

    # Nulos restantes
    total_nulos = df.isnull().sum().sum()
    if total_nulos > 0:
        print(f'  ⚠️  Nulos restantes: {total_nulos}')
    else:
        print('  ✅ No quedan valores nulos')

    # Shape final
    print(f'\n  Shape final del dataset limpio:')
    print(f'    Filas   : {df.shape[0]:,}')
    print(f'    Columnas: {df.shape[1]}')
    print(f'\n  Columnas finales:')
    for col in df.columns:
        print(f'    • {col:<40} {df[col].dtype}')


# ─────────────────────────────────────────────────────────────
#  PASO 8 — Grafico de distribución post-limpieza
# ─────────────────────────────────────────────────────────────

def grafico_distribucion_numericas(df):
    """Boxplots de variables numéricas tras la limpieza."""
    vars_num = ['tenure', 'MonthlyCharges', 'TotalCharges']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, col in enumerate(vars_num):
        sns.boxplot(
            data=df, y=col,
            color='#3498db',
            ax=axes[i]
        )
        axes[i].set_title(f'{col}')
        axes[i].set_ylabel(col)

    plt.suptitle('Distribución de Variables Numéricas — Post Limpieza',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    ruta = os.path.join(OUTPUT_DIR, '02_numericas_post_limpieza.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  ✅ Gráfico guardado: {ruta}')


# ─────────────────────────────────────────────────────────────
#  PASO 9 — Guardar dataset limpio
# ─────────────────────────────────────────────────────────────

def guardar_dataset(df, nombre='telco_churn_limpio.csv'):
    """Guarda el dataset procesado en data/ para la siguiente etapa."""
    ruta = os.path.join(DATA_DIR, nombre)
    df.to_csv(ruta, index=False)
    print(f'\n  ✅ Dataset limpio guardado: {ruta}')
    print(f'     Listo para → src/features/feature_engineering.py')


# =============================================================
#  RESUMEN
# =============================================================

def resumen_final(df):
    print('\n' + '=' * 60)
    print('  RESUMEN — LIMPIEZA COMPLETADA')
    print('=' * 60)
    print('  PASOS EJECUTADOS:')
    print('  1. ✅ Eliminado customerID')
    print('  2. ✅ TotalCharges convertida a float')
    print('  3. ✅ Nulos imputados (mediana por Contract)')
    print('  4. ✅ Churn codificado: Yes=1 / No=0')
    print('  5. ✅ Label Encoding aplicado (13 columnas)')
    print('  6. ✅ One-Hot Encoding aplicado (InternetService, PaymentMethod)')
    print('  7. ✅ Sin columnas object ni nulos restantes')
    print()
    print(f'  Shape final: {df.shape[0]:,} filas × {df.shape[1]} columnas')
    print()
    print('  SIGUIENTE PASO → python src/features/feature_engineering.py')
    print('=' * 60)


# =============================================================
#  MAIN
# =============================================================

def main():
    print('\n🚀 INICIANDO LIMPIEZA DEL DATASET...\n')

    df = cargar_dataset('telco_churn.csv')

    df = eliminar_columnas(df)
    df = corregir_tipos(df)
    df = tratar_nulos(df)
    df = codificar_target(df)
    df = aplicar_label_encoding(df)
    df = aplicar_ohe(df)

    verificar_resultado(df)
    grafico_distribucion_numericas(df)
    guardar_dataset(df)
    resumen_final(df)


if __name__ == '__main__':
    main()