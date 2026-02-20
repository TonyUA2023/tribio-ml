import os, json
import numpy as np
import pandas as pd
from joblib import load, dump

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, precision_recall_curve
)

PROCESSED_DIR = "data/processed"
ART_DIR = "artifacts"
OUT_MODEL_PATH = os.path.join(ART_DIR, "model_module2.joblib")

def print_metrics(y_true, y_pred, y_proba=None, title=""):
    print(f"\n=== {title} ===")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("F1:", round(f1_score(y_true, y_pred, zero_division=0), 4))
    print("CM:\n", confusion_matrix(y_true, y_pred))
    if y_proba is not None:
        print("ROC-AUC:", round(roc_auc_score(y_true, y_proba), 4))
        print("PR-AUC:", round(average_precision_score(y_true, y_proba), 4))
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))

def best_threshold_for_f1(y_true, y_proba):
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    f1s = (2 * prec * rec) / (prec + rec + 1e-9)
    idx = int(np.nanargmax(f1s))
    best_thr = float(thr[idx]) if idx < len(thr) else 0.5
    return best_thr, float(f1s[idx])

def main():
    X_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "m2_X_train.parquet"))
    X_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "m2_X_test.parquet"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "m2_y_train.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "m2_y_test.npy"))

    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Churn rate train:", round(y_train.mean(), 4), "test:", round(y_test.mean(), 4))

    # Baseline 1: clase mayoritaria
    maj = int(np.bincount(y_train).argmax())
    y_pred_maj = np.full_like(y_test, maj)
    print_metrics(y_test, y_pred_maj, title="Baseline 1: Clase mayoritaria")

    # Baseline 2: regla R (Recency) simple
    if "days_since_last_order" in X_test.columns:
        r = X_test["days_since_last_order"].astype(float).to_numpy()
        thr_r = np.median(X_train["days_since_last_order"].astype(float))
        y_pred_r = (r >= thr_r).astype(int)
        proba_r = (r - r.min()) / (r.max() - r.min() + 1e-9)
        print_metrics(y_test, y_pred_r, y_proba=proba_r, title=f"Baseline 2: Recency>=mediana({thr_r:.1f})")

    prep_bundle = load(os.path.join(ART_DIR, "preprocess_module2.joblib"))
    preprocess = prep_bundle["preprocess"]
    target = prep_bundle.get("target", "will_churn_30d")

    # Modelo 1: LR
    lr = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", n_jobs=-1)
    pipe_lr = Pipeline([("prep", preprocess), ("model", lr)])
    pipe_lr.fit(X_train, y_train)

    proba_lr = pipe_lr.predict_proba(X_test)[:, 1]
    thr_lr, f1_lr = best_threshold_for_f1(y_test, proba_lr)
    pred_lr = (proba_lr >= thr_lr).astype(int)

    print_metrics(y_test, pred_lr, y_proba=proba_lr, title=f"Modelo 1: Logistic (thr={thr_lr:.3f}, F1={f1_lr:.3f})")

    # Modelo 2: RF
    rf = RandomForestClassifier(
        n_estimators=500, n_jobs=-1, random_state=42,
        class_weight="balanced_subsample",
        min_samples_split=4, min_samples_leaf=2
    )
    pipe_rf = Pipeline([("prep", preprocess), ("model", rf)])
    pipe_rf.fit(X_train, y_train)

    proba_rf = pipe_rf.predict_proba(X_test)[:, 1]
    pred_rf = (proba_rf >= 0.5).astype(int)

    print_metrics(y_test, pred_rf, y_proba=proba_rf, title="Modelo 2: RandomForest (thr=0.5)")

    # Selección por PR-AUC
    pr_lr = average_precision_score(y_test, proba_lr)
    pr_rf = average_precision_score(y_test, proba_rf)

    if pr_rf > pr_lr:
        best = {"model_name": "random_forest", "pipeline": pipe_rf, "threshold": 0.5, "proba": proba_rf}
    else:
        best = {"model_name": "logistic_regression", "pipeline": pipe_lr, "threshold": float(thr_lr), "proba": proba_lr}

    bundle = {
        "model_name": best["model_name"],
        "pipeline": best["pipeline"],
        "threshold": float(best["threshold"]),
        "target": target,
        "metrics": {
            "pr_auc": float(average_precision_score(y_test, best["proba"])),
            "roc_auc": float(roc_auc_score(y_test, best["proba"])),
        }
    }

    dump(bundle, OUT_MODEL_PATH)

    with open(os.path.join(ART_DIR, "model_module2_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_name": bundle["model_name"],
            "threshold": bundle["threshold"],
            "target": bundle["target"],
            "metrics": bundle["metrics"],
        }, f, ensure_ascii=False, indent=2)

    print("\n✅ Guardado:", OUT_MODEL_PATH)
    print("Modelo elegido:", bundle["model_name"], "| thr:", round(bundle["threshold"], 3))

if __name__ == "__main__":
    main()