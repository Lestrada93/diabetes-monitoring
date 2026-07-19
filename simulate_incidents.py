"""
simulate_incidents.py
Simulación de incidentes críticos para el modelo de predicción de diabetes.
Registra cada incidente en MLflow y ejecuta el runbook correspondiente.
"""

import mlflow
import numpy as np
import pandas as pd
import urllib.request
import time
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, roc_auc_score, f1_score

import runbooks

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("incidents.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

SLO_RECALL_MIN = 0.80
SLO_AUC_MIN    = 0.75
SLO_LATENCIA   = 500  # ms

# ── Helpers ────────────────────────────────────────────────────────────────
def cargar_y_preparar():
    URL  = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    COLS = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
    urllib.request.urlretrieve(URL, "diabetes.csv")
    df = pd.read_csv("diabetes.csv", header=None, names=COLS)
    for col in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
        df[col] = df[col].replace(0, df[col][df[col] != 0].median())
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_sc, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, max_depth=8,
                                   random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluar(model, X_test, y_test, umbral=0.20):
    t0    = time.time()
    proba = model.predict_proba(X_test)[:, 1]
    lat   = (time.time() - t0) * 1000
    pred  = (proba >= umbral).astype(int)
    tn = ((pred==0)&(y_test==0)).sum()
    fp = ((pred==1)&(y_test==0)).sum()
    return {
        "recall":        round(recall_score(y_test, pred), 4),
        "roc_auc":       round(roc_auc_score(y_test, proba), 4),
        "f1":            round(f1_score(y_test, pred), 4),
        "especificidad": round(tn/(tn+fp) if (tn+fp)>0 else 0, 4),
        "latencia_ms":   round(lat, 2),
    }

# ── Incidente 1: Degradación del modelo ────────────────────────────────────
def incidente_degradacion():
    print("\n" + "="*60)
    print("  INCIDENTE 1: Degradación del modelo por data drift")
    print("="*60)

    mlflow.set_experiment("diabetes-monitoring")
    model, X_test, y_test = cargar_y_preparar()

    # Simular data drift: ruido gaussiano en los datos de entrada
    np.random.seed(99)
    X_degradado = X_test + np.random.normal(0, 3.0, X_test.shape)

    with mlflow.start_run(run_name="incidente_degradacion_modelo"):
        mlflow.set_tags({
            "tipo_incidente": "degradacion_modelo",
            "causa":          "data_drift",
            "severidad":      "ALTA",
            "timestamp":      datetime.now().isoformat()
        })

        # Métricas ANTES
        m_antes = evaluar(model, X_test, y_test)
        logger.info(f"Métricas ANTES del incidente: {m_antes}")
        mlflow.log_metrics({f"antes_{k}": v for k, v in m_antes.items()})

        # Métricas DURANTE (degradado)
        m_durante = evaluar(model, X_degradado, y_test)
        logger.warning(f"Métricas DURANTE incidente: {m_durante}")
        mlflow.log_metrics({f"durante_{k}": v for k, v in m_durante.items()})

        # Detectar violación SLO
        resultado = None
        if m_durante["recall"] < SLO_RECALL_MIN or m_durante["roc_auc"] < SLO_AUC_MIN:
            logger.error(f"SLO VIOLATION: Recall {m_durante['recall']} < {SLO_RECALL_MIN} o AUC {m_durante['roc_auc']} < {SLO_AUC_MIN}")
            mlflow.set_tag("slo_status", "VIOLATION")

            # Ejecutar runbook
            print("\n  → Ejecutando Runbook: Reentrenamiento del modelo")
            resultado = runbooks.runbook_reentrenamiento(model, X_test, y_test)
            mlflow.log_metrics({f"despues_{k}": v for k, v in resultado.items()})
            mlflow.set_tag("accion", "reentrenamiento")
            mlflow.set_tag("resolucion", "OK" if resultado["recall"] >= SLO_RECALL_MIN else "PENDIENTE")
        else:
            mlflow.set_tag("slo_status", "OK")
            resultado = m_durante

        print(f"\n  Antes   → Recall: {m_antes['recall']} | AUC: {m_antes['roc_auc']}")
        print(f"  Durante → Recall: {m_durante['recall']} | AUC: {m_durante['roc_auc']}")
        print(f"  Después → Recall: {resultado['recall']} | AUC: {resultado['roc_auc']}")

# ── Incidente 2: Alta latencia ──────────────────────────────────────────────
def incidente_latencia():
    print("\n" + "="*60)
    print("  INCIDENTE 2: Incremento de latencia en predicciones")
    print("="*60)

    mlflow.set_experiment("diabetes-monitoring")
    model, X_test, y_test = cargar_y_preparar()

    with mlflow.start_run(run_name="incidente_alta_latencia"):
        mlflow.set_tags({
            "tipo_incidente": "alta_latencia",
            "causa":          "carga_excesiva",
            "severidad":      "MEDIA",
            "timestamp":      datetime.now().isoformat()
        })

        # Simular alta latencia añadiendo delay artificial
        logger.warning("Simulando alta latencia en el servicio...")
        t0    = time.time()
        proba = model.predict_proba(X_test)[:, 1]
        time.sleep(0.6)  # Simular sobrecarga
        latencia_simulada = (time.time() - t0) * 1000

        pred = (proba >= 0.20).astype(int)
        tn   = ((pred==0)&(y_test==0)).sum()
        fp   = ((pred==1)&(y_test==0)).sum()

        metricas = {
            "recall":        round(recall_score(y_test, pred), 4),
            "roc_auc":       round(roc_auc_score(y_test, proba), 4),
            "latencia_ms":   round(latencia_simulada, 2),
            "especificidad": round(tn/(tn+fp) if (tn+fp)>0 else 0, 4),
        }
        mlflow.log_metrics(metricas)

        if metricas["latencia_ms"] > SLO_LATENCIA:
            logger.error(f"SLO VIOLATION: Latencia {metricas['latencia_ms']}ms > {SLO_LATENCIA}ms")
            mlflow.set_tag("slo_status", "VIOLATION")

            print("\n  → Ejecutando Runbook: Escalamiento del servicio")
            resultado = runbooks.runbook_escalamiento()
            mlflow.set_tag("accion", "escalamiento")
            mlflow.set_tag("resolucion", resultado["status"])

        print(f"\n  Latencia detectada: {metricas['latencia_ms']:.1f}ms")
        print(f"  SLO máximo:         {SLO_LATENCIA}ms")
        print(f"  Recall:             {metricas['recall']}")
        print(f"  Resolución:         {resultado['status']}")

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    incidente_degradacion()
    incidente_latencia()
    print("\n✅ Simulación de incidentes completada.")
    print("   Ejecuta: mlflow ui  para ver los resultados")
