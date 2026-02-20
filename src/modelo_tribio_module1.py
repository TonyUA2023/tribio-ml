# src/modelo_tribio_module1.py
import os
import json
import numpy as np
import pandas as pd
from joblib import load, dump

from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, precision_recall_curve
)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


PROCESSED_DIR = os.getenv("TRIBIO_PROCESSED_DIR", "data/processed")
ART_DIR = os.getenv("TRIBIO_ARTIFACTS_DIR", "artifacts")
OUT_MODEL_PATH = os.getenv("TRIBIO_MODEL_OUT", os.path.join(ART_DIR, "model_module1.joblib"))

def load_processed():
    X_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_train.parquet"))
    X_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_test.parquet"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    return X_train, X_test, y_train, y_test

def print_metrics(y_true, y_pred, y_proba=None, title=""):
    print(f"\n=== {title} ===")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("F1:", round(f1_score(y_true, y_pred, zero_division=0), 4))
    print("CM:\n", confusion_matrix(y_true, y_pred))

    if y_proba is not None:
        try:
            print("ROC-AUC:", round(roc_auc_score(y_true, y_proba), 4))
        except Exception:
            print("ROC-AUC: (no disponible)")
        print("PR-AUC:", round(average_precision_score(y_true, y_proba), 4))

    print(classification_report(y_true, y_pred, digits=3, zero_division=0))

def best_threshold_for_f1(y_true, y_proba):
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    f1s = (2 * prec * rec) / (prec + rec + 1e-9)
    idx = int(np.nanargmax(f1s))
    # thr tiene length-1 vs prec/rec; protegemos índice
    best_thr = float(thr[idx]) if idx < len(thr) else 0.5
    best_f1 = float(f1s[idx])
    return best_thr, best_f1

def main():
    # 0) Cargar datos procesados
    X_train, X_test, y_train, y_test = load_processed()

    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Positive rate train:", round(y_train.mean(), 4), "test:", round(y_test.mean(), 4))

    # 1) Baseline 1: Clase mayoritaria
    majority = int(np.bincount(y_train).argmax())
    y_pred_major = np.full_like(y_test, fill_value=majority)
    print_metrics(y_test, y_pred_major, title="Baseline 1: Clase mayoritaria")

    # 2) Baseline 2: Regla simple por historial (ventas pasadas)
    # Usamos product_sales_30d como proxy: si vendió algo en 30d -> predice que venderá
    if "product_sales_30d" in X_test.columns:
        score_hist = X_test["product_sales_30d"].astype(float)
        # umbral: >0 suele ser razonable
        y_pred_hist = (score_hist > 0).astype(int).to_numpy()
        # para AUC necesitamos probas: normalizamos a [0,1]
        proba_hist = (score_hist - score_hist.min()) / (score_hist.max() - score_hist.min() + 1e-9)
        print_metrics(y_test, y_pred_hist, y_proba=proba_hist, title="Baseline 2: Historial ventas (sales_30d>0)")
    else:
        print("\nBaseline 2 omitido: no existe product_sales_30d en X_test")

    # 3) Cargar preprocess (OneHot + Imputer + Scaling) generado en Fase 2
    prep_bundle = load(os.path.join(ART_DIR, "preprocess_module1.joblib"))
    preprocess = prep_bundle["preprocess"]
    cutoff_date = prep_bundle.get("cutoff_date", "N/A")
    target = prep_bundle.get("target", "will_sell_next_7d")

    # 4) Modelo 1: Regresión logística (robusto, explicable)
    lr = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1
    )

    pipe_lr = Pipeline([
        ("prep", preprocess),
        ("model", lr)
    ])

    pipe_lr.fit(X_train, y_train)

    proba_lr = pipe_lr.predict_proba(X_test)[:, 1]
    pred_lr_05 = (proba_lr >= 0.5).astype(int)

    print_metrics(y_test, pred_lr_05, y_proba=proba_lr, title="Modelo 1: LogisticRegression (umbral=0.5)")

    # Umbral óptimo para F1 (como en tu ejemplo de clase, pero mejorado)
    best_thr, best_f1 = best_threshold_for_f1(y_test, proba_lr)
    pred_lr_opt = (proba_lr >= best_thr).astype(int)
    print_metrics(y_test, pred_lr_opt, y_proba=proba_lr, title=f"Modelo 1: LogisticRegression (umbral óptimo={best_thr:.4f}, F1={best_f1:.4f})")

    # 5) Modelo 2: Random Forest (baseline no lineal)
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample"
    )

    pipe_rf = Pipeline([
        ("prep", preprocess),
        ("model", rf)
    ])

    pipe_rf.fit(X_train, y_train)

    proba_rf = pipe_rf.predict_proba(X_test)[:, 1]
    pred_rf = (proba_rf >= 0.5).astype(int)

    print_metrics(y_test, pred_rf, y_proba=proba_rf, title="Modelo 2: RandomForest (umbral=0.5)")

    # 6) Selección del mejor modelo (por PR-AUC, y luego F1)
    pr_lr = average_precision_score(y_test, proba_lr)
    pr_rf = average_precision_score(y_test, proba_rf)

    if pr_rf > pr_lr:
        best_model_name = "random_forest"
        best_pipe = pipe_rf
        best_proba = proba_rf
        best_threshold = 0.5
    else:
        best_model_name = "logistic_regression"
        best_pipe = pipe_lr
        best_proba = proba_lr
        best_threshold = best_thr  # usamos el umbral optimizado

    # 7) Guardar artefactos para nube (modelo + umbral + metadata)
    os.makedirs(ART_DIR, exist_ok=True)

    bundle = {
        "model_name": best_model_name,
        "pipeline": best_pipe,
        "threshold": float(best_threshold),
        "target": target,
        "cutoff_date": cutoff_date,
        "metrics": {
            "pr_auc": float(average_precision_score(y_test, best_proba)),
            "roc_auc": float(roc_auc_score(y_test, best_proba)),
        }
    }

    dump(bundle, OUT_MODEL_PATH)

    # también guardamos un JSON legible (útil para despliegue / logs)
    meta_path = os.path.join(ART_DIR, "model_module1_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": best_model_name,
            "threshold": float(best_threshold),
            "target": target,
            "cutoff_date": cutoff_date,
            "metrics": bundle["metrics"]
        }, f, ensure_ascii=False, indent=2)

    print("\n✅ Guardado:")
    print("-", OUT_MODEL_PATH)
    print("-", meta_path)
    print("Modelo elegido:", best_model_name, "| threshold:", round(best_threshold, 4))

if __name__ == "__main__":
    main()