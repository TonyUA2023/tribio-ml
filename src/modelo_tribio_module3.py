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
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix

ART_DIR = "artifacts"
PROCESSED_DIR = "data/processed"

MODEL_OUT = os.path.join(ART_DIR, "model_module3.joblib")
META_OUT  = os.path.join(ART_DIR, "model_module3_metadata.json")

TARGET = "high_conversion_session"

def find_best_threshold(y_true, proba):
    best_thr = 0.5
    best_val = -1
    for thr in np.linspace(0.05, 0.95, 181):
        pred = (proba >= thr).astype(int)
        val = f1_score(y_true, pred)
        if val > best_val:
            best_val = val
            best_thr = thr
    return best_thr, best_val

def main():
    os.makedirs(ART_DIR, exist_ok=True)

    X_train = pd.read_parquet(f"{PROCESSED_DIR}/m3_X_train.parquet")
    X_test  = pd.read_parquet(f"{PROCESSED_DIR}/m3_X_test.parquet")
    y_train = np.load(f"{PROCESSED_DIR}/m3_y_train.npy")
    y_test  = np.load(f"{PROCESSED_DIR}/m3_y_test.npy")

    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    model = LogisticRegression(max_iter=3000, class_weight="balanced")

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:,1]
    thr, f1best = find_best_threshold(y_test, proba)
    pred = (proba >= thr).astype(int)

    metrics = {
        "f1_best": float(f1best),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "conversion_rate": float(y_test.mean())
    }

    bundle = {
        "pipeline": pipe,
        "threshold": float(thr),
        "model_name": "LogisticRegression",
        "target": TARGET,
        "metrics": metrics
    }

    dump(bundle, MODEL_OUT)

    with open(META_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Modelo M3 entrenado")
    print("ROC-AUC:", round(metrics["roc_auc"],4))

if __name__ == "__main__":
    main()