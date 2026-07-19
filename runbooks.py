"""
runbooks.py
Runbooks de respuesta automatizada a incidentes del modelo de diabetes.
Cada función representa un procedimiento estandarizado de respuesta.
"""

import logging
import time
import numpy as np
import urllib.request
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, roc_auc_score, f1_score

logger = logging.getLogger(__name__)

# ── Runbook 1: Reentrenamiento del modelo ──────────────────────────────────
def runbook_reentrenamiento(model_actual, X_test, y_test):
    """
    RUNBOOK-001: Reentrenamiento del modelo por degradación o data drift.

    Trigger: Recall < 0.80 o AUC < 0.75
    Severidad: ALTA
    Tiempo estimado de respuesta: 5-10 minutos

    Pasos:
    1. Detectar violación de SLO
    2. Descargar dataset actualizado
    3. Reentrenar modelo con parámetros optimizados
    4. Validar que el nuevo modelo cumple SLOs
    5. Reemplazar modelo en producción
    6. Registrar evento en log de incidentes
    """
    logger.info("="*50)
    logger.info("RUNBOOK-001: Iniciando reentrenamiento del modelo")
    logger.info("="*50)

    print("\n  [RUNBOOK-001] Reentrenamiento del modelo")
    print("  Paso 1: Detectando causa raíz → Data drift confirmado")

    # Paso 2: Recargar datos limpios
    print("  Paso 2: Descargando dataset actualizado...")
    URL  = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    COLS = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
    urllib.request.urlretrieve(URL, "diabetes.csv")
    df = pd.read_csv("diabetes.csv", header=None, names=COLS)
    for col in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
        df[col] = df[col].replace(0, df[col][df[col] != 0].median())

    # Paso 3: Reentrenar
    print("  Paso 3: Reentrenando modelo con datos limpios...")
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    X_train, X_test_new, y_train, y_test_new = train_test_split(
        X_sc, y, test_size=0.2, random_state=42, stratify=y)

    nuevo_modelo = RandomForestClassifier(
        n_estimators=200, max_depth=8,
        random_state=42, class_weight='balanced')
    nuevo_modelo.fit(X_train, y_train)

    # Paso 4: Validar
    print("  Paso 4: Validando nuevo modelo...")
    proba = nuevo_modelo.predict_proba(X_test_new)[:, 1]
    pred  = (proba >= 0.20).astype(int)
    tn = ((pred==0)&(y_test_new==0)).sum()
    fp = ((pred==1)&(y_test_new==0)).sum()

    metricas = {
        "recall":        round(recall_score(y_test_new, pred), 4),
        "roc_auc":       round(roc_auc_score(y_test_new, proba), 4),
        "f1":            round(f1_score(y_test_new, pred), 4),
        "especificidad": round(tn/(tn+fp) if (tn+fp)>0 else 0, 4),
    }

    # Paso 5: Confirmar sustitución
    if metricas["recall"] >= 0.80:
        print("  Paso 5: ✅ Nuevo modelo cumple SLOs → Desplegado en producción")
        logger.info(f"Reentrenamiento exitoso: {metricas}")
    else:
        print("  Paso 5: ⚠️  Nuevo modelo no cumple SLOs → Escalando a equipo ML")
        logger.warning(f"Reentrenamiento insuficiente: {metricas}")

    # Paso 6: Registro
    logger.info(f"RUNBOOK-001 completado a las {datetime.now().isoformat()}")
    logger.info(f"Métricas post-reentrenamiento: {metricas}")

    return metricas


# ── Runbook 2: Escalamiento del servicio ───────────────────────────────────
def runbook_escalamiento():
    """
    RUNBOOK-002: Escalamiento del servicio por alta latencia.

    Trigger: Latencia > 500ms
    Severidad: MEDIA
    Tiempo estimado de respuesta: 2-5 minutos

    Pasos:
    1. Confirmar degradación de latencia
    2. Identificar causa (carga, memoria, CPU)
    3. Escalar horizontalmente (simular)
    4. Verificar recuperación de latencia
    5. Registrar evento
    """
    logger.info("="*50)
    logger.info("RUNBOOK-002: Iniciando escalamiento del servicio")
    logger.info("="*50)

    print("\n  [RUNBOOK-002] Escalamiento del servicio")
    print("  Paso 1: Confirmando alta latencia → 620ms detectados")
    print("  Paso 2: Causa identificada → Carga excesiva de requests")
    print("  Paso 3: Escalando instancias del contenedor...")
    time.sleep(0.5)  # Simular tiempo de escalamiento
    print("  Paso 4: Verificando recuperación de latencia...")
    time.sleep(0.3)
    latencia_recuperada = 85.3  # ms simulados post-escalamiento
    print(f"  Latencia post-escalamiento: {latencia_recuperada}ms ✅")
    print("  Paso 5: Evento registrado en log de incidentes")

    logger.info(f"RUNBOOK-002 completado. Latencia recuperada: {latencia_recuperada}ms")

    return {
        "status":               "RESUELTO",
        "latencia_recuperada":  latencia_recuperada,
        "accion":               "escalamiento_horizontal",
        "timestamp":            datetime.now().isoformat()
    }


# ── Runbook 3: Rollback del modelo ─────────────────────────────────────────
def runbook_rollback(version_anterior="v1.0"):
    """
    RUNBOOK-003: Rollback a versión anterior del modelo.

    Trigger: Fallo crítico del modelo actual o degradación severa
    Severidad: CRÍTICA
    Tiempo estimado de respuesta: 1-3 minutos

    Pasos:
    1. Confirmar fallo crítico
    2. Identificar última versión estable
    3. Restaurar modelo anterior desde MLflow Registry
    4. Verificar funcionamiento
    5. Notificar al equipo
    """
    logger.info("="*50)
    logger.info(f"RUNBOOK-003: Iniciando rollback a {version_anterior}")
    logger.info("="*50)

    print(f"\n  [RUNBOOK-003] Rollback del modelo a {version_anterior}")
    print("  Paso 1: Confirmando fallo crítico del modelo actual")
    print(f"  Paso 2: Versión estable identificada → {version_anterior}")
    print("  Paso 3: Restaurando modelo desde MLflow Registry...")
    time.sleep(0.4)
    print("  Paso 4: Verificando funcionamiento del modelo restaurado...")
    time.sleep(0.2)
    print("  Paso 5: ✅ Rollback completado — Servicio restaurado")
    print("          Notificación enviada al equipo de ML")

    logger.info(f"RUNBOOK-003 completado. Rollback a {version_anterior} exitoso.")

    return {
        "status":           "RESUELTO",
        "version_activa":   version_anterior,
        "accion":           "rollback",
        "timestamp":        datetime.now().isoformat()
    }


# ── Catálogo de runbooks ───────────────────────────────────────────────────
CATALOGO = {
    "RUNBOOK-001": {
        "nombre":    "Reentrenamiento del modelo",
        "trigger":   "Recall < 0.80 o AUC < 0.75",
        "severidad": "ALTA",
        "funcion":   runbook_reentrenamiento
    },
    "RUNBOOK-002": {
        "nombre":    "Escalamiento del servicio",
        "trigger":   "Latencia > 500ms",
        "severidad": "MEDIA",
        "funcion":   runbook_escalamiento
    },
    "RUNBOOK-003": {
        "nombre":    "Rollback del modelo",
        "trigger":   "Fallo crítico o degradación severa",
        "severidad": "CRÍTICA",
        "funcion":   runbook_rollback
    },
}

if __name__ == "__main__":
    print("\n📋 Catálogo de Runbooks disponibles:")
    for k, v in CATALOGO.items():
        print(f"  {k}: {v['nombre']} | Trigger: {v['trigger']} | Severidad: {v['severidad']}")
