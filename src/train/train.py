# =============================================================
#  src/train/train.py
#  Proyecto : Predicción de Abandono de Clientes (Churn)
#  Etapa    : 4 — Entrenamiento de Modelos
#  Ejecutar : python src/train/train.py
#
#  Propósito:
#    Entrenar, optimizar y comparar 5 modelos de ML más un
#    ensamble, para determinar cuál predice mejor el churn.
#    Guarda cada modelo entrenado como .pkl para reutilizarlo.
# =============================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import time

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import (RandomForestClassifier,
                                     GradientBoostingClassifier,
                                     VotingClassifier)
from xgboost                 import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     roc_auc_score)
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', palette='Set2')

# ── Rutas del proyecto ────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'figuras')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42


# =============================================================
#  FUNCIONES
# =============================================================

def cargar_datos():
    """
    Carga los 4 conjuntos generados por feature_engineering.py.
    X_train ya tiene SMOTE aplicado.
    X_test refleja datos reales sin modificar.
    """
    print('=' * 60)
    print('  ENTRENAMIENTO DE MODELOS')
    print('=' * 60)

    X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
    X_test  = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).squeeze()
    y_test  = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).squeeze()

    print(f'  X_train : {X_train.shape}  (con SMOTE)')
    print(f'  X_test  : {X_test.shape}   (datos reales)')
    print(f'  Churn en train → 0: {(y_train==0).sum():,} | 1: {(y_train==1).sum():,}')
    print(f'  Churn en test  → 0: {(y_test==0).sum():,}  | 1: {(y_test==1).sum():,}')

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────
#  PASO 1 — Definir modelos e hiperparámetros
# ─────────────────────────────────────────────────────────────

def definir_modelos():
    """
    Define los 5 modelos con sus grillas de hiperparámetros
    para optimización mediante GridSearchCV.

    ¿Por qué estos hiperparámetros?
    - LR  : C controla regularización (evita overfitting)
    - DT  : max_depth limita profundidad del árbol
    - RF  : n_estimators = cantidad de árboles en el bosque
    - GB  : learning_rate controla cuánto aporta cada árbol
    - XGB : scale_pos_weight maneja desbalance residual
    """
    modelos = {

        'Logistic Regression': {
            'modelo': LogisticRegression(
                random_state=SEED,
                max_iter=1000,
                class_weight='balanced'
            ),
            'params': {
                'C'      : [0.01, 0.1, 1, 10],
                'solver' : ['lbfgs', 'liblinear'],
            }
        },

        'Decision Tree': {
            'modelo': DecisionTreeClassifier(
                random_state=SEED,
                class_weight='balanced'
            ),
            'params': {
                'max_depth'        : [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'criterion'        : ['gini', 'entropy'],
            }
        },

        'Random Forest': {
            'modelo': RandomForestClassifier(
                random_state=SEED,
                class_weight='balanced',
                n_jobs=-1
            ),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth'   : [5, 10, 20, None],
                'max_features': ['sqrt', 'log2'],
            }
        },

        'Gradient Boosting': {
            'modelo': GradientBoostingClassifier(
                random_state=SEED
            ),
            'params': {
                'n_estimators' : [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth'    : [3, 5],
            }
        },

        'XGBoost': {
            'modelo': XGBClassifier(
                random_state=SEED,
                eval_metric='logloss',
                verbosity=0,
                use_label_encoder=False
            ),
            'params': {
                'n_estimators'     : [100, 200, 300],
                'learning_rate'    : [0.05, 0.1, 0.2],
                'max_depth'        : [3, 5, 7],
                'scale_pos_weight' : [1, 2],   # maneja desbalance residual
            }
        },
    }

    print(f'\n── PASO 1: Modelos definidos ───────────────────────────')
    for nombre in modelos:
        n_combinaciones = 1
        for vals in modelos[nombre]['params'].values():
            n_combinaciones *= len(vals)
        print(f'  • {nombre:<22} → {n_combinaciones} combinaciones de hiperparámetros')

    return modelos


# ─────────────────────────────────────────────────────────────
#  PASO 2 — Entrenar y optimizar cada modelo
# ─────────────────────────────────────────────────────────────

def entrenar_modelos(modelos, X_train, y_train):
    """
    Usa GridSearchCV con validación cruzada k=5 para encontrar
    los mejores hiperparámetros de cada modelo.

    scoring='roc_auc' porque es la métrica principal del proyecto
    (más informativa que accuracy en datos desbalanceados).
    """
    print('\n── PASO 2: Entrenamiento con GridSearchCV (k=5) ────────')
    print('  (esto puede tardar unos minutos...)\n')

    resultados = {}

    for nombre, config in modelos.items():
        print(f'  🔄 Entrenando: {nombre}')
        inicio = time.time()

        grid = GridSearchCV(
            estimator  = config['modelo'],
            param_grid = config['params'],
            cv         = 5,
            scoring    = 'roc_auc',
            n_jobs     = -1,
            verbose    = 0
        )
        grid.fit(X_train, y_train)

        tiempo = time.time() - inicio
        print(f'     ✅ Listo en {tiempo:.1f}s')
        print(f'     Mejores parámetros : {grid.best_params_}')
        print(f'     AUC-ROC (CV train) : {grid.best_score_:.4f}\n')

        resultados[nombre] = {
            'modelo'      : grid.best_estimator_,
            'best_params' : grid.best_params_,
            'auc_cv'      : grid.best_score_,
        }

    return resultados


# ─────────────────────────────────────────────────────────────
#  PASO 3 — Modelo ensamblado (Voting)
# ─────────────────────────────────────────────────────────────

def crear_ensamble(resultados, X_train, y_train):
    """
    Combina los 3 mejores modelos en un VotingClassifier.
    voting='soft' usa probabilidades en lugar de votos directos,
    lo que suele dar mejores resultados con AUC-ROC.
    """
    print('── PASO 3: Modelo Ensamblado (Voting Classifier) ───────')

    # Tomar los 3 modelos con mayor AUC en CV
    top3 = sorted(
        resultados.items(),
        key=lambda x: x[1]['auc_cv'],
        reverse=True
    )[:3]

    estimadores = [(nombre, config['modelo']) for nombre, config in top3]
    nombres_top3 = [nombre for nombre, _ in top3]

    print(f'  Top 3 modelos seleccionados:')
    for nombre, config in top3:
        print(f'    • {nombre:<22} AUC-CV: {config["auc_cv"]:.4f}')

    ensamble = VotingClassifier(
        estimators=estimadores,
        voting='soft',
        n_jobs=-1
    )
    ensamble.fit(X_train, y_train)

    print(f'\n  ✅ VotingClassifier entrenado con: {nombres_top3}')

    resultados['Voting Ensemble'] = {
        'modelo'     : ensamble,
        'best_params': {'voting': 'soft', 'top3': nombres_top3},
        'auc_cv'     : None,
    }

    return resultados


# ─────────────────────────────────────────────────────────────
#  PASO 4 — Evaluar en test y comparar
# ─────────────────────────────────────────────────────────────

def evaluar_modelos(resultados, X_test, y_test):
    """
    Evalúa todos los modelos sobre el test set (datos reales).
    Calcula: Accuracy, Precision, Recall, F1, AUC-ROC.

    Se prioriza Recall y AUC-ROC porque el costo de no detectar
    un cliente en riesgo (falso negativo) es mayor que el de
    una retención innecesaria (falso positivo).
    """
    print('\n── PASO 4: Evaluación en Test Set ──────────────────────')

    filas = []

    for nombre, config in resultados.items():
        modelo  = config['modelo']
        y_pred  = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1]

        fila = {
            'Modelo'    : nombre,
            'Accuracy'  : accuracy_score(y_test, y_pred),
            'Precision' : precision_score(y_test, y_pred, zero_division=0),
            'Recall'    : recall_score(y_test, y_pred, zero_division=0),
            'F1-Score'  : f1_score(y_test, y_pred, zero_division=0),
            'AUC-ROC'   : roc_auc_score(y_test, y_proba),
        }
        filas.append(fila)

        # Guardar predicciones en el config para usarlas en evaluate.py
        resultados[nombre]['y_pred']  = y_pred
        resultados[nombre]['y_proba'] = y_proba

    df_metricas = pd.DataFrame(filas).set_index('Modelo')
    df_metricas = df_metricas.sort_values('AUC-ROC', ascending=False)

    print('\n  📊 Tabla comparativa de modelos:')
    print(f'\n  {"Modelo":<22} {"Accuracy":>9} {"Precision":>10} '
          f'{"Recall":>8} {"F1":>8} {"AUC-ROC":>9}')
    print(f'  {"-"*70}')
    for idx, row in df_metricas.iterrows():
        print(f'  {idx:<22} {row["Accuracy"]:>9.4f} {row["Precision"]:>10.4f} '
              f'{row["Recall"]:>8.4f} {row["F1-Score"]:>8.4f} {row["AUC-ROC"]:>9.4f}')

    mejor = df_metricas.index[0]
    print(f'\n  🏆 Mejor modelo: {mejor}  '
          f'(AUC-ROC: {df_metricas.loc[mejor, "AUC-ROC"]:.4f})')

    return df_metricas, resultados


# ─────────────────────────────────────────────────────────────
#  PASO 5 — Gráfico comparativo de métricas
# ─────────────────────────────────────────────────────────────

def grafico_comparativa(df_metricas):
    """Gráfico de barras comparando todas las métricas por modelo."""
    metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    df_plot  = df_metricas[metricas].reset_index()
    df_melt  = df_plot.melt(id_vars='Modelo', var_name='Métrica', value_name='Valor')

    plt.figure(figsize=(14, 6))
    sns.barplot(
        data=df_melt,
        x='Modelo', y='Valor', hue='Métrica',
        palette='Set2'
    )
    plt.title('Comparativa de Métricas por Modelo', fontsize=14, fontweight='bold')
    plt.xlabel('Modelo')
    plt.ylabel('Valor')
    plt.xticks(rotation=15, ha='right')
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()

    ruta = os.path.join(OUTPUT_DIR, '04_comparativa_modelos.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  ✅ Gráfico guardado: {ruta}')


def grafico_auc_ranking(df_metricas):
    """Ranking visual de modelos por AUC-ROC."""
    df_sorted = df_metricas.sort_values('AUC-ROC')

    colores = ['#e74c3c' if i == len(df_sorted) - 1 else '#3498db'
               for i in range(len(df_sorted))]

    plt.figure(figsize=(10, 5))
    barras = plt.barh(
        df_sorted.index,
        df_sorted['AUC-ROC'],
        color=colores,
        edgecolor='white',
        height=0.5
    )
    for barra, val in zip(barras, df_sorted['AUC-ROC']):
        plt.text(val + 0.002, barra.get_y() + barra.get_height() / 2,
                 f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

    plt.title('Ranking de Modelos por AUC-ROC', fontsize=14, fontweight='bold')
    plt.xlabel('AUC-ROC')
    plt.xlim(0, 1.05)
    plt.tight_layout()

    ruta = os.path.join(OUTPUT_DIR, '04_ranking_auc_roc.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✅ Gráfico guardado: {ruta}')


# ─────────────────────────────────────────────────────────────
#  PASO 6 — Guardar modelos
# ─────────────────────────────────────────────────────────────

def guardar_modelos(resultados):
    """
    Guarda cada modelo entrenado como archivo .pkl.
    joblib es más eficiente que pickle para modelos sklearn.
    """
    print('\n── PASO 6: Guardar modelos entrenados ──────────────────')

    nombres_archivo = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree'      : 'decision_tree.pkl',
        'Random Forest'      : 'random_forest.pkl',
        'Gradient Boosting'  : 'gradient_boosting.pkl',
        'XGBoost'            : 'xgboost.pkl',
        'Voting Ensemble'    : 'voting_ensemble.pkl',
    }

    for nombre, config in resultados.items():
        nombre_archivo = nombres_archivo.get(nombre, f'{nombre}.pkl')
        ruta = os.path.join(MODELS_DIR, nombre_archivo)
        joblib.dump(config['modelo'], ruta)
        print(f'  ✅ {nombre:<22} → {nombre_archivo}')

    # Guardar también la tabla de métricas
    ruta_metricas = os.path.join(MODELS_DIR, 'metricas_modelos.csv')
    print(f'\n  (Las métricas se guardan en evaluate.py)')


# ─────────────────────────────────────────────────────────────
#  RESUMEN
# ─────────────────────────────────────────────────────────────

def resumen_final(df_metricas):
    mejor        = df_metricas.index[0]
    auc_mejor    = df_metricas.loc[mejor, 'AUC-ROC']
    recall_mejor = df_metricas.loc[mejor, 'Recall']
    f1_mejor     = df_metricas.loc[mejor, 'F1-Score']

    print('\n' + '=' * 60)
    print('  RESUMEN — ENTRENAMIENTO COMPLETADO')
    print('=' * 60)
    print(f'  Modelos entrenados : 5 individuales + 1 ensamble')
    print(f'  Optimización       : GridSearchCV k=5, scoring=AUC-ROC')
    print()
    print(f'  🏆 MEJOR MODELO: {mejor}')
    print(f'     AUC-ROC : {auc_mejor:.4f}')
    print(f'     Recall  : {recall_mejor:.4f}')
    print(f'     F1-Score: {f1_mejor:.4f}')
    print()
    print('  ARCHIVOS GENERADOS:')
    print('  models/ → .pkl de cada modelo')
    print('  outputs/figuras/ → gráficos comparativos')
    print()
    print('  SIGUIENTE PASO → python src/evaluation/evaluate.py')
    print('=' * 60)


# =============================================================
#  MAIN
# =============================================================

def main():
    print('\n🚀 INICIANDO ENTRENAMIENTO DE MODELOS...\n')

    # 1. Cargar datos
    X_train, X_test, y_train, y_test = cargar_datos()

    # 2. Definir modelos
    modelos = definir_modelos()

    # 3. Entrenar con GridSearchCV
    resultados = entrenar_modelos(modelos, X_train, y_train)

    # 4. Ensamble con top 3
    resultados = crear_ensamble(resultados, X_train, y_train)

    # 5. Evaluar en test
    df_metricas, resultados = evaluar_modelos(resultados, X_test, y_test)

    # 6. Gráficos
    print('\n📊 GENERANDO GRÁFICOS...')
    grafico_comparativa(df_metricas)
    grafico_auc_ranking(df_metricas)

    # 7. Guardar modelos
    guardar_modelos(resultados)

    # 8. Resumen
    resumen_final(df_metricas)


if __name__ == '__main__':
    main()