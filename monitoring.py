"""
monitoring.py
Sistema de monitorización del modelo de predicción de diabetes.
Integra métricas, logs y trazas con MLflow.
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pickle
import time
import logging
import urllib.request
from datetime import datetime
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ── Configuración de logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("monitoring.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── SLOs definidos ─────────────────────────────────────────────────────────
SLO = {
    "recall_min":        0.80,   # Recall mínimo aceptable
    "especificidad_min": 0.70,   # Especificidad mínima
    "auc_min":           0.75,   # AUC mínimo
    "latencia_max_ms":   500,    # Latencia máxima en ms
    "error_budget":      0.05    # 5% de margen de error permitido
}

# ── Cargar dataset ─────────────────────────────────────────────────────────
def cargar_datos():
    URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    COLS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    urllib.request.urlretrieve(URL, "diabetes.csv")
    df = pd.read_csv("diabetes.csv", header=None, names=COLS)
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        df[col] = df[col].replace(0, df[col][df[col] != 0].median())
    return df

# ── Entrenar modelo ────────────────────────────────────────────────────────
def entrenar_modelo(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(
        n_estimators=200, max_depth=8,
        random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model, scaler, X_test, y_test

# ── Calcular métricas ──────────────────────────────────────────────────────
def calcular_metricas(model, X_test, y_test, umbral=0.20):
    inicio = time.time()
    proba  = model.predict_proba(X_test)[:, 1]
    latencia_ms = (time.time() - inicio) * 1000

    y_pred = (proba >= umbral).astype(int)
    tn = ((y_pred == 0) & (y_test == 0)).sum()
    fp = ((y_pred == 1) & (y_test == 0)).sum()

    metricas = {
        "accuracy":       round(accuracy_score(y_test, y_pred), 4),
        "precision":      round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":         round(recall_score(y_test, y_pred), 4),
        "f1_score":       round(f1_score(y_test, y_pred), 4),
        "roc_auc":        round(roc_auc_score(y_test, proba), 4),
        "especificidad":  round(tn / (tn + fp) if (tn + fp) > 0 else 0, 4),
        "latencia_ms":    round(latencia_ms, 2),
        "umbral":         umbral,
        "n_muestras":     len(y_test),
        "n_positivos":    int(y_pred.sum()),
        "tasa_positivos": round(y_pred.mean(), 4),
    }
    return metricas

# ── Verificar SLOs ─────────────────────────────────────────────────────────
def verificar_slos(metricas):
    violaciones = []
    if metricas["recall"] < SLO["recall_min"]:
        violaciones.append(f"RECALL {metricas['recall']} < SLO {SLO['recall_min']}")
    if metricas["especificidad"] < SLO["especificidad_min"]:
        violaciones.append(f"ESPECIFICIDAD {metricas['especificidad']} < SLO {SLO['especificidad_min']}")
    if metricas["roc_auc"] < SLO["auc_min"]:
        violaciones.append(f"AUC {metricas['roc_auc']} < SLO {SLO['auc_min']}")
    if metricas["latencia_ms"] > SLO["latencia_max_ms"]:
        violaciones.append(f"LATENCIA {metricas['latencia_ms']}ms > SLO {SLO['latencia_max_ms']}ms")
    return violaciones

# ── Experimento MLflow principal ───────────────────────────────────────────
def run_monitoreo(nombre_run="monitoreo_baseline", umbral=0.20, degradar=False):
    logger.info(f"Iniciando run de monitoreo: {nombre_run}")

    df = cargar_datos()
    model, scaler, X_test, y_test = entrenar_modelo(df)

    # Simular degradación si se solicita
    if degradar:
        logger.warning("Simulando degradación del modelo — añadiendo ruido a los datos de prueba")
        np.random.seed(42)
        X_test = X_test + np.random.normal(0, 2.5, X_test.shape)

    metricas = calcular_metricas(model, X_test, y_test, umbral)
    violaciones = verificar_slos(metricas)

    mlflow.set_experiment("diabetes-monitoring")

    with mlflow.start_run(run_name=nombre_run):
        # ── Tags ──────────────────────────────────────────────────────────
        mlflow.set_tags({
            "modelo":    "RandomForest",
            "dataset":   "PimaIndiansDiabetes",
            "umbral":    str(umbral),
            "degradado": str(degradar),
            "autor":     "Luis Alonso Estrada Uribe",
            "timestamp": datetime.now().isoformat()
        })

        # ── Parámetros ────────────────────────────────────────────────────
        mlflow.log_params({
            "n_estimators":   200,
            "max_depth":      8,
            "umbral":         umbral,
            "test_size":      0.2,
            "class_weight":   "balanced",
            "slo_recall_min": SLO["recall_min"],
            "slo_auc_min":    SLO["auc_min"],
            "error_budget":   SLO["error_budget"]
        })

        # ── Métricas ──────────────────────────────────────────────────────
        mlflow.log_metrics(metricas)
        mlflow.log_metric("slo_violaciones", len(violaciones))
        mlflow.log_metric("error_budget_consumido",
                          max(0, SLO["recall_min"] - metricas["recall"]))

        # ── Logs de violaciones ───────────────────────────────────────────
        if violaciones:
            logger.error(f"SLO VIOLATIONS detectadas: {violaciones}")
            for v in violaciones:
                logger.error(f"  ⚠️  {v}")
            mlflow.set_tag("status", "SLO_VIOLATION")
        else:
            logger.info("Todos los SLOs cumplidos ✅")
            mlflow.set_tag("status", "OK")

        # ── Guardar modelo ────────────────────────────────────────────────
        mlflow.sklearn.log_model(model, "random_forest_model")

        # ── Resumen en consola ────────────────────────────────────────────
        print(f"\n{'='*55}")
        print(f"  Run: {nombre_run}")
        print(f"{'='*55}")
        for k, v in metricas.items():
            print(f"  {k:<25}: {v}")
        if violaciones:
            print(f"\n  ⚠️  VIOLACIONES SLO:")
            for v in violaciones:
                print(f"    - {v}")
        else:
            print(f"\n  ✅ Todos los SLOs cumplidos")
        print(f"{'='*55}\n")

        return metricas, violaciones

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔵 Run 1: Monitoreo baseline (normal)")
    run_monitoreo("baseline_normal", umbral=0.20, degradar=False)

    print("\n🔴 Run 2: Monitoreo con degradación simulada")
    run_monitoreo("modelo_degradado", umbral=0.20, degradar=True)

    print("\n🟡 Run 3: Umbral estándar 0.50 (comparación)")
    run_monitoreo("umbral_estandar_050", umbral=0.50, degradar=False)

    print("\n✅ Monitoreo completado.")
