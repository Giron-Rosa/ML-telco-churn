# =============================================================
#  src/features/feature_engineering.py
#  Proyecto : Predicción de Abandono de Clientes (Churn)
#  Etapa    : 3 — Ingeniería de Características
#  Ejecutar : python src/features/feature_engineering.py
# =============================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', palette='Set2')

# ── Rutas del proyecto ────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'figuras')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Columnas numéricas a escalar ──────────────────────────
COLS_ESCALAR = ['tenure', 'MonthlyCharges', 'TotalCharges',
                'avg_charge_per_month', 'charge_per_service']

# Semilla para reproducibilidad
SEED = 42


# =============================================================
#  FUNCIONES
# =============================================================

def cargar_dataset(nombre='telco_churn_limpio.csv'):
    """Carga el dataset limpio generado por clean.py."""
    ruta = os.path.join(DATA_DIR, nombre)
    df = pd.read_csv(ruta)
    print('=' * 60)
    print('  INGENIERÍA DE CARACTERÍSTICAS')
    print('=' * 60)
    print(f'  Dataset cargado: {nombre}')
    print(f'  Filas: {df.shape[0]:,}  |  Columnas: {df.shape[1]}')
    return df


# ─────────────────────────────────────────────────────────────
#  PASO 1 — Crear nuevas variables (features derivadas)
# ─────────────────────────────────────────────────────────────

def crear_features(df):
    """
    Crea 4 variables derivadas que capturan patrones
    de comportamiento relevantes para predecir el churn.
    """
    print('\n── PASO 1: Crear nuevas variables ──────────────────────')

    # 1. Número de servicios contratados
    #    Más servicios → mayor retención (más atado al proveedor)
    servicios = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies'
    ]
    df['num_services'] = df[servicios].sum(axis=1)
    print(f'  ✅ num_services         → suma de {len(servicios)} servicios activos')
    print(f'     Rango: {df["num_services"].min()} – {df["num_services"].max()}')
    print(f'     Media: {df["num_services"].mean():.2f}')

    # 2. Cargo promedio por mes de antigüedad
    #    Captura la carga financiera real del cliente
    df['avg_charge_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)
    print(f'\n  ✅ avg_charge_per_month → TotalCharges / (tenure + 1)')
    print(f'     Media: ${df["avg_charge_per_month"].mean():.2f}')

    # 3. Indicador de cliente nuevo (tenure ≤ 6 meses)
    #    Clientes nuevos tienen mayor probabilidad de abandono
    df['is_new_customer'] = (df['tenure'] <= 6).astype(int)
    nuevos = df['is_new_customer'].sum()
    print(f'\n  ✅ is_new_customer      → 1 si tenure ≤ 6 meses')
    print(f'     Clientes nuevos: {nuevos:,}  ({nuevos/len(df)*100:.1f}%)')

    # 4. Cargo por servicio contratado
    #    Mide si el precio es percibido como justo por servicios recibidos
    df['charge_per_service'] = df['MonthlyCharges'] / (df['num_services'] + 1)
    print(f'\n  ✅ charge_per_service   → MonthlyCharges / (num_services + 1)')
    print(f'     Media: ${df["charge_per_service"].mean():.2f}')

    print(f'\n  Columnas totales tras feature engineering: {df.shape[1]}')
    return df


# ─────────────────────────────────────────────────────────────
#  PASO 2 — Separar features y target
# ─────────────────────────────────────────────────────────────

def separar_features_target(df):
    """Separa X (features) e y (target: Churn)."""
    print('\n── PASO 2: Separar features y target ───────────────────')

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    print(f'  ✅ X (features) : {X.shape[1]} columnas')
    print(f'  ✅ y (target)   : Churn  →  0={( y==0).sum():,}  |  1={(y==1).sum():,}')
    return X, y


# ─────────────────────────────────────────────────────────────
#  PASO 3 — Split train / test
# ─────────────────────────────────────────────────────────────

def dividir_datos(X, y):
    """
    Divide en 80% entrenamiento y 20% prueba.
    stratify=y garantiza la misma proporción de churn
    en ambos conjuntos (importante con datos desbalanceados).
    """
    print('\n── PASO 3: División Train / Test ───────────────────────')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=SEED,
        stratify=y           # mantiene proporción de clases
    )

    print(f'  ✅ Train : {X_train.shape[0]:,} filas  ({X_train.shape[0]/len(X)*100:.0f}%)')
    print(f'  ✅ Test  : {X_test.shape[0]:,} filas  ({X_test.shape[0]/len(X)*100:.0f}%)')
    print(f'\n  Distribución Churn en Train:')
    print(f'    No Churn (0): {(y_train==0).sum():,}')
    print(f'    Churn    (1): {(y_train==1).sum():,}')
    print(f'\n  Distribución Churn en Test:')
    print(f'    No Churn (0): {(y_test==0).sum():,}')
    print(f'    Churn    (1): {(y_test==1).sum():,}')

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────
#  PASO 4 — Escalado de variables numéricas
# ─────────────────────────────────────────────────────────────

def escalar_variables(X_train, X_test):
    """
    Aplica StandardScaler SOLO a las columnas numéricas continuas.
    IMPORTANTE: el scaler se entrena con train y se aplica a test,
    nunca al revés (evita data leakage).
    """
    print('\n── PASO 4: Escalado de variables numéricas ─────────────')

    # Filtrar solo las columnas que existen en el dataset
    cols_a_escalar = [c for c in COLS_ESCALAR if c in X_train.columns]

    scaler = StandardScaler()

    # Fit SOLO con train → transform a ambos
    X_train[cols_a_escalar] = scaler.fit_transform(X_train[cols_a_escalar])
    X_test[cols_a_escalar]  = scaler.transform(X_test[cols_a_escalar])

    print(f'  ✅ StandardScaler aplicado a: {cols_a_escalar}')
    print(f'     Fit con train → transform en train y test')
    print(f'\n  Estadísticas tras escalado (train):')
    print(f'  {"Columna":<25} {"Media":>8} {"Std":>8}')
    print(f'  {"-"*43}')
    for col in cols_a_escalar:
        print(f'  {col:<25} {X_train[col].mean():>8.4f} {X_train[col].std():>8.4f}')

    # Guardar el scaler para usarlo en predicciones futuras
    ruta_scaler = os.path.join(MODELS_DIR, 'scaler.pkl')
    joblib.dump(scaler, ruta_scaler)
    print(f'\n  ✅ Scaler guardado: {ruta_scaler}')

    return X_train, X_test, scaler


# ─────────────────────────────────────────────────────────────
#  PASO 5 — SMOTE (balanceo de clases)
# ─────────────────────────────────────────────────────────────

def aplicar_smote(X_train, y_train):
    """
    Aplica SMOTE SOLO al conjunto de entrenamiento.
    NUNCA al test — el test debe reflejar la realidad.
    SMOTE genera ejemplos sintéticos de la clase minoritaria (Churn=1)
    interpolando entre ejemplos existentes.
    """
    print('\n── PASO 5: SMOTE — Balanceo de clases ──────────────────')

    print(f'  Antes del SMOTE:')
    print(f'    No Churn (0): {(y_train==0).sum():,}')
    print(f'    Churn    (1): {(y_train==1).sum():,}')
    print(f'    Ratio        : {(y_train==1).sum()/(y_train==0).sum():.2f}')

    smote = SMOTE(random_state=SEED)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print(f'\n  Después del SMOTE:')
    print(f'    No Churn (0): {(y_train_bal==0).sum():,}')
    print(f'    Churn    (1): {(y_train_bal==1).sum():,}')
    print(f'    Ratio        : {(y_train_bal==1).sum()/(y_train_bal==0).sum():.2f}')
    print(f'    Total filas  : {len(X_train_bal):,}')
    print(f'\n  ✅ SMOTE aplicado solo a train (test sin modificar)')

    return X_train_bal, y_train_bal


# ─────────────────────────────────────────────────────────────
#  PASO 6 — Gráficos
# ─────────────────────────────────────────────────────────────

def grafico_nuevas_features(df):
    """Visualiza las 4 features nuevas vs Churn."""
    features_nuevas = [
        'num_services', 'avg_charge_per_month',
        'is_new_customer', 'charge_per_service'
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colores = {'0': '#2ecc71', '1': '#e74c3c'}

    for i, col in enumerate(features_nuevas):
        if col == 'is_new_customer':
            tasa = df.groupby(col)['Churn'].mean() * 100
            barras = axes[i].bar(
                ['Cliente establecido\n(tenure > 6)', 'Cliente nuevo\n(tenure ≤ 6)'],
                tasa.values,
                color=['#2ecc71', '#e74c3c'],
                edgecolor='white', width=0.5
            )
            for barra, val in zip(barras, tasa.values):
                axes[i].text(
                    barra.get_x() + barra.get_width() / 2,
                    barra.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold'
                )
            axes[i].set_title('Tasa de Churn: Cliente Nuevo vs Establecido')
            axes[i].set_ylabel('Churn (%)')
            axes[i].set_ylim(0, tasa.max() * 1.3)
        else:
            # Convertir Churn a string para que coincida con las claves del palette
            df_plot = df[[col, 'Churn']].copy()
            df_plot['Churn'] = df_plot['Churn'].astype(str)
            sns.boxplot(
                data=df_plot, x='Churn', y=col,
                palette={'0': '#2ecc71', '1': '#e74c3c'},
                order=['0', '1'],
                ax=axes[i]
            )
            axes[i].set_title(f'{col} por Churn')
            axes[i].set_xticklabels(['No Churn (0)', 'Churn (1)'])

    plt.suptitle('Nuevas Features vs. Churn', fontsize=14, fontweight='bold')
    plt.tight_layout()

    ruta = os.path.join(OUTPUT_DIR, '03_nuevas_features_vs_churn.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  ✅ Gráfico guardado: {ruta}')


def grafico_smote(y_train_orig, y_train_bal):
    """Comparativa antes y después del SMOTE."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colores = ['#2ecc71', '#e74c3c']

    datos = [
        (y_train_orig, 'Antes del SMOTE\n(Train original)'),
        (y_train_bal,  'Después del SMOTE\n(Train balanceado)')
    ]

    for ax, (y, titulo) in zip(axes, datos):
        counts = pd.Series(y).value_counts().sort_index()
        barras = ax.bar(
            ['No Churn (0)', 'Churn (1)'],
            counts.values,
            color=colores, edgecolor='white', width=0.5
        )
        for barra, val in zip(barras, counts.values):
            ax.text(
                barra.get_x() + barra.get_width() / 2,
                barra.get_height() + 20,
                f'{val:,}', ha='center', fontsize=11, fontweight='bold'
            )
        ax.set_title(titulo, fontsize=12)
        ax.set_ylabel('Número de registros')
        ax.set_ylim(0, counts.max() * 1.2)

    plt.suptitle('Efecto del SMOTE en el conjunto de entrenamiento',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    ruta = os.path.join(OUTPUT_DIR, '03_smote_balanceo.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✅ Gráfico guardado: {ruta}')


# ─────────────────────────────────────────────────────────────
#  PASO 7 — Guardar datasets finales
# ─────────────────────────────────────────────────────────────

def guardar_datasets(X_train, X_test, y_train, y_test):
    """
    Guarda los 4 conjuntos listos para el entrenamiento.
    El train ya tiene SMOTE aplicado.
    """
    print('\n── PASO 7: Guardar datasets finales ────────────────────')

    archivos = {
        'X_train.csv': X_train,
        'X_test.csv' : X_test,
        'y_train.csv': pd.DataFrame(y_train, columns=['Churn']),
        'y_test.csv' : pd.DataFrame(y_test,  columns=['Churn']),
    }

    for nombre, datos in archivos.items():
        ruta = os.path.join(DATA_DIR, nombre)
        datos.to_csv(ruta, index=False)
        print(f'  ✅ Guardado: {ruta}  ({datos.shape})')

    print(f'\n  Listos para → python src/train/train.py')


# =============================================================
#  RESUMEN
# =============================================================

def resumen_final(X_train, X_test, y_train, y_test):
    print('\n' + '=' * 60)
    print('  RESUMEN — FEATURE ENGINEERING COMPLETADO')
    print('=' * 60)
    print('  PASOS EJECUTADOS:')
    print('  1. ✅ 4 nuevas features creadas')
    print('       num_services, avg_charge_per_month,')
    print('       is_new_customer, charge_per_service')
    print('  2. ✅ Split 80/20 con estratificación')
    print('  3. ✅ StandardScaler aplicado (sin data leakage)')
    print('  4. ✅ SMOTE aplicado solo en train')
    print()
    print(f'  CONJUNTOS FINALES:')
    print(f'    X_train (balanceado): {X_train.shape}')
    print(f'    X_test  (real)      : {X_test.shape}')
    print(f'    y_train             : Churn=0: {(y_train==0).sum():,} | Churn=1: {(y_train==1).sum():,}')
    print(f'    y_test              : Churn=0: {(y_test==0).sum():,}  | Churn=1: {(y_test==1).sum():,}')
    print()
    print('  SIGUIENTE PASO → python src/train/train.py')
    print('=' * 60)


# =============================================================
#  MAIN
# =============================================================

def main():
    print('\n🚀 INICIANDO FEATURE ENGINEERING...\n')

    # 1. Cargar dataset limpio
    df = cargar_dataset('telco_churn_limpio.csv')

    # 2. Crear nuevas features
    df = crear_features(df)

    # 3. Separar X e y
    X, y = separar_features_target(df)

    # 4. Split train/test
    X_train, X_test, y_train, y_test = dividir_datos(X, y)

    # 5. Escalar variables numéricas
    X_train, X_test, scaler = escalar_variables(X_train, X_test)

    # 6. SMOTE — guardar y_train original para el gráfico
    y_train_original = y_train.copy()
    X_train, y_train = aplicar_smote(X_train, y_train)

    # 7. Gráficos
    print('\n📊 GENERANDO GRÁFICOS...')
    grafico_nuevas_features(df)
    grafico_smote(y_train_original, y_train)

    # 8. Guardar datasets finales
    guardar_datasets(X_train, X_test, y_train, y_test)

    # 9. Resumen
    resumen_final(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()