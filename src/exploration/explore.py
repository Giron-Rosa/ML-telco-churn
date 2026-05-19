# =============================================================
#  src/exploration/explore.py
#  Proyecto : Predicción de Abandono de Clientes (Churn)
#  Etapa    : 1 — Exploración del Dataset (EDA)
#  Ejecutar : python src/exploration/explore.py
# =============================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')           # modo headless: guarda figuras sin pantalla
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ── Configuración general ─────────────────────────────────
sns.set_theme(style='whitegrid', palette='Set2')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.titlesize']   = 13
plt.rcParams['axes.titleweight'] = 'bold'

# ── Rutas del proyecto ────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'outputs', 'figuras')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================
#  FUNCIONES
# =============================================================

def cargar_dataset(nombre_archivo='telco_churn.csv'):
    """Carga el dataset desde data/ y muestra info básica."""
    ruta = os.path.join(DATA_DIR, nombre_archivo)
    df = pd.read_csv(ruta)
    print('=' * 55)
    print('  DATASET CARGADO')
    print('=' * 55)
    print(f'  Archivo  : {nombre_archivo}')
    print(f'  Filas    : {df.shape[0]:,}')
    print(f'  Columnas : {df.shape[1]}')
    print('=' * 55)
    return df


def clasificar_variables(df):
    """Clasifica columnas en numéricas, categóricas y target."""
    col_id          = ['customerID']
    col_target      = ['Churn']
    col_numericas   = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    col_categoricas = df.select_dtypes(include=['object']).columns.tolist()
    col_categoricas = [c for c in col_categoricas if c not in col_id + col_target]

    print('\n📋 CLASIFICACIÓN DE VARIABLES')
    print(f'  🔢 Numéricas   ({len(col_numericas)}): {col_numericas}')
    print(f'  🔤 Categóricas ({len(col_categoricas)}): {col_categoricas}')
    print(f'  🎯 Target      : {col_target[0]}')
    print(f'  🗑️  Descartar   : {col_id[0]}')

    return col_numericas, col_categoricas, col_id, col_target


def analizar_nulos(df):
    """Detecta NaN reales y espacios vacíos en TotalCharges."""
    print('\n📋 ANÁLISIS DE VALORES NULOS')
    print('-' * 40)

    # NaN reales
    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0]
    if nulos.empty:
        print('  ✅ No se detectaron NaN explícitos')
    else:
        print('  ⚠️  NaN detectados:')
        print(nulos.to_string())

    # Espacios vacíos en TotalCharges (bug conocido del dataset)
    vacios = (df['TotalCharges'].str.strip() == '').sum()
    print(f'\n  ⚠️  TotalCharges con espacio vacío: {vacios} registros')
    print('     → Se convertirá a NaN en la etapa de limpieza')

    # Duplicados
    duplicados = df.duplicated().sum()
    print(f'\n  Filas duplicadas: {duplicados}')


def estadisticas_descriptivas(df, col_numericas):
    """Imprime estadísticas de variables numéricas y categóricas."""
    print('\n📊 ESTADÍSTICAS — VARIABLES NUMÉRICAS')
    print('-' * 40)
    print(df[col_numericas].describe().round(2).to_string())


def estadisticas_categoricas(df, col_categoricas):
    """Imprime frecuencias de variables categóricas."""
    print('\n📊 FRECUENCIAS — VARIABLES CATEGÓRICAS')
    print('-' * 40)
    for col in col_categoricas:
        freq = df[col].value_counts(normalize=True).mul(100).round(1)
        print(f'\n  [{col}] — {df[col].nunique()} valores únicos')
        for val, pct in freq.items():
            print(f'      {val:<35} {pct}%')


def grafico_churn(df):
    """Genera y guarda gráfico de distribución de la variable objetivo."""
    churn_counts = df['Churn'].value_counts()
    churn_pct    = df['Churn'].value_counts(normalize=True).mul(100).round(1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colores = ['#2ecc71', '#e74c3c']

    # Barras
    axes[0].bar(churn_counts.index, churn_counts.values,
                color=colores, width=0.5, edgecolor='white')
    for i, (val, pct) in enumerate(zip(churn_counts.values, churn_pct.values)):
        axes[0].text(i, val + 50, f'{val:,}\n({pct}%)',
                     ha='center', fontsize=11, fontweight='bold')
    axes[0].set_title('Distribución de Churn (conteo)')
    axes[0].set_xlabel('Churn')
    axes[0].set_ylabel('Número de clientes')
    axes[0].set_ylim(0, churn_counts.max() * 1.2)

    # Torta
    axes[1].pie(
        churn_pct.values,
        labels=[f'No Churn\n{churn_pct["No"]}%', f'Churn\n{churn_pct["Yes"]}%'],
        colors=colores,
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        textprops={'fontsize': 12}
    )
    axes[1].set_title('Proporción Churn vs. No Churn')

    plt.suptitle('Variable Objetivo: Churn', fontsize=14, fontweight='bold')
    plt.tight_layout()

    ruta = os.path.join(OUTPUT_DIR, '01_distribucion_churn.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✅ Guardado: {ruta}')


def grafico_numericas(df, col_numericas):
    """Histogramas y boxplots de variables numéricas vs Churn."""
    vars_num = ['tenure', 'MonthlyCharges', 'TotalCharges']
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for i, col in enumerate(vars_num):
        sns.histplot(
            data=df, x=col, hue='Churn',
            kde=True, bins=30,
            palette={'Yes': '#e74c3c', 'No': '#2ecc71'},
            alpha=0.6, ax=axes[0, i]
        )
        axes[0, i].set_title(f'Distribución: {col}')

        sns.boxplot(
            data=df, x='Churn', y=col,
            palette={'Yes': '#e74c3c', 'No': '#2ecc71'},
            ax=axes[1, i]
        )
        axes[1, i].set_title(f'Boxplot: {col} por Churn')

    plt.suptitle('Variables Numéricas vs. Churn', fontsize=14, fontweight='bold')
    plt.tight_layout()

    ruta = os.path.join(OUTPUT_DIR, '01_numericas_vs_churn.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✅ Guardado: {ruta}')


def grafico_categoricas(df):
    """Tasa de churn por cada variable categórica importante."""
    vars_cat = [
        'Contract', 'InternetService', 'PaymentMethod',
        'OnlineSecurity', 'TechSupport', 'PaperlessBilling'
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, col in enumerate(vars_cat):
        tasa = df.groupby(col)['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).sort_values(ascending=False)

        barras = axes[i].bar(
            tasa.index, tasa.values,
            color=sns.color_palette('Set2', len(tasa)),
            edgecolor='white'
        )
        for barra, val in zip(barras, tasa.values):
            axes[i].text(
                barra.get_x() + barra.get_width() / 2,
                barra.get_height() + 0.5,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold'
            )
        axes[i].set_title(f'Tasa de Churn por {col}')
        axes[i].set_ylabel('Churn (%)')
        axes[i].set_ylim(0, tasa.max() * 1.25)
        axes[i].tick_params(axis='x', rotation=15)

    plt.suptitle('Tasa de Churn por Variable Categórica',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    ruta = os.path.join(OUTPUT_DIR, '01_categoricas_vs_churn.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✅ Guardado: {ruta}')


def grafico_correlacion(df):
    """Heatmap de correlación entre variables numéricas y Churn."""
    df_corr = df[['tenure', 'MonthlyCharges', 'TotalCharges',
                  'SeniorCitizen', 'Churn']].copy()
    df_corr['Churn'] = df_corr['Churn'].map({'Yes': 1, 'No': 0})

    # Convertir TotalCharges a numérica si aún no lo está
    df_corr['TotalCharges'] = pd.to_numeric(df_corr['TotalCharges'], errors='coerce')

    corr = df_corr.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, mask=mask, square=True,
                linewidths=0.5, cbar_kws={'shrink': 0.8})
    plt.title('Matriz de Correlación', fontweight='bold')
    plt.tight_layout()

    ruta = os.path.join(OUTPUT_DIR, '01_correlacion.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✅ Guardado: {ruta}')

    print('\n  Correlaciones con Churn:')
    print(corr['Churn'].sort_values(ascending=False).round(3).to_string())


def resumen_final(df):
    """Imprime el resumen final del EDA."""
    print('\n' + '=' * 55)
    print('  RESUMEN FINAL — EXPLORACIÓN')
    print('=' * 55)
    print(f'  Total registros     : {df.shape[0]:,}')
    print(f'  Total columnas      : {df.shape[1]}')
    print(f'  Duplicados          : {df.duplicated().sum()}')
    vacios = (df['TotalCharges'].str.strip() == '').sum()
    print(f'  Nulos TotalCharges  : {vacios}')
    print(f'  Tasa Churn (Yes)    : {(df["Churn"]=="Yes").mean()*100:.1f}%')
    print()
    print('  HALLAZGOS:')
    print('  • TotalCharges tiene espacios vacíos → tratar en limpieza')
    print('  • Dataset desbalanceado ~26.5% churn → usar SMOTE')
    print('  • Contract y tenure: mayor correlación con churn')
    print('  • Fiber optic tiene mayor tasa de abandono que DSL')
    print()
    print('  GRÁFICOS GENERADOS en outputs/figuras/:')
    print('  • 01_distribucion_churn.png')
    print('  • 01_numericas_vs_churn.png')
    print('  • 01_categoricas_vs_churn.png')
    print('  • 01_correlacion.png')
    print()
    print('  SIGUIENTE PASO → python src/preprocessing/clean.py')
    print('=' * 55)


# =============================================================
#  MAIN
# =============================================================

def main():
    print('\n🚀 INICIANDO EXPLORACIÓN DEL DATASET...\n')

    # 1. Cargar
    df = cargar_dataset('telco_churn.csv')

    # 2. Clasificar variables
    col_numericas, col_categoricas, col_id, col_target = clasificar_variables(df)

    # 3. Nulos e inconsistencias
    analizar_nulos(df)

    # 4. Estadísticas
    estadisticas_descriptivas(df, col_numericas)
    estadisticas_categoricas(df, col_categoricas)

    # 5. Gráficos
    print('\n📊 GENERANDO GRÁFICOS...')
    grafico_churn(df)
    grafico_numericas(df, col_numericas)
    grafico_categoricas(df)
    grafico_correlacion(df)

    # 6. Resumen
    resumen_final(df)


if __name__ == '__main__':
    main()