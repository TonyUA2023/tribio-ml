import os
import json
import numpy as np
import pandas as pd

from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix
)

ART_DIR = "artifacts"
PROCESSED_DIR = "data/processed"

MODEL_OUT = os.path.join(ART_DIR, "model_module1.joblib")
META_OUT  = os.path.join(ART_DIR, "model_module1_metadata.json")

TARGET = "will_sell_next_7d"  # (solo informativo)


def find_best_threshold(y_true, proba, metric="f1"):
    # Busca threshold que maximiza F1 (por defecto)
    best_thr = 0.5
    best_val = -1.0

    for thr in np.linspace(0.05, 0.95, 181):
        pred = (proba >= thr).astype(int)
        val = f1_score(y_true, pred, zero_division=0)
        if val > best_val:
            best_val = val
            best_thr = float(thr)

    return best_thr, best_val


def main():
    os.makedirs(ART_DIR, exist_ok=True)

    # -----------------------
    # 1) Cargar data preparada
    # -----------------------
    X_train = pd.read_parquet(f"{PROCESSED_DIR}/X_train.parquet")
    X_test  = pd.read_parquet(f"{PROCESSED_DIR}/X_test.parquet")
    y_train = np.load(f"{PROCESSED_DIR}/y_train.npy").astype(int)
    y_test  = np.load(f"{PROCESSED_DIR}/y_test.npy").astype(int)

    # -----------------------
    # 2) Definir columnas num/cat
    # -----------------------
    # Nota: esto se adapta a lo que exista realmente
    num_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    # -----------------------
    # 3) Preprocess robusto (para nube)
    # -----------------------
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop"
    )

    # -----------------------
    # 4) Modelo baseline fuerte (LR balanceada)
    # -----------------------
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", clf)
    ])

    # -----------------------
    # 5) Entrenar
    # -----------------------
    pipe.fit(X_train, y_train)

    # -----------------------
    # 6) Evaluar + threshold óptimo
    # -----------------------
    proba_test = pipe.predict_proba(X_test)[:, 1]
    best_thr, best_f1 = find_best_threshold(y_test, proba_test)

    pred_best = (proba_test >= best_thr).astype(int)

    metrics = {
        "f1_best": float(best_f1),
        "roc_auc": float(roc_auc_score(y_test, proba_test)),
        "pr_auc": float(average_precision_score(y_test, proba_test)),
        "confusion_matrix": confusion_matrix(y_test, pred_best).tolist(),
        "pos_rate_test": float(y_test.mean()),
        "pred_rate_test": float(pred_best.mean()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features_input": int(X_train.shape[1]),
        "model_name": "LogisticRegression(class_weight=balanced)",
    }

    # -----------------------
    # 7) Guardar bundle para dashboard
    # -----------------------
    bundle = {
        "pipeline": pipe,
        "threshold": float(best_thr),
        "model_name": metrics["model_name"],
        "target": TARGET,
        "metrics": metrics,
        "feature_columns": X_train.columns.tolist(),
        "num_cols": num_cols,
        "cat_cols": cat_cols
    }

    dump(bundle, MODEL_OUT)

    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("✅ Entrenamiento completado Módulo 1")
    print(f"📦 Modelo: {MODEL_OUT}")
    print(f"🧾 Métricas: {META_OUT}")
    print("🎯 Threshold óptimo:", round(best_thr, 4))
    print("F1(best):", round(metrics["f1_best"], 4))
    print("ROC-AUC:", round(metrics["roc_auc"], 4))
    print("PR-AUC:", round(metrics["pr_auc"], 4))
    print("CM:\n", np.array(metrics["confusion_matrix"]))


if __name__ == "__main__":
    main()