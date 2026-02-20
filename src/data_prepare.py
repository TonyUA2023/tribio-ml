# src/data_prepare.py
import os
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

RAW_PATH = os.getenv("TRIBIO_RAW_PATH", "data/raw/module1_product_week_raw.parquet")
OUT_DIR = os.getenv("TRIBIO_PROCESSED_DIR", "data/processed")
ART_DIR = os.getenv("TRIBIO_ARTIFACTS_DIR", "artifacts")

TARGET_CLASSIF = os.getenv("TRIBIO_TARGET", "will_sell_next_7d")
TIME_COL = "week_start"

# Columnas que NUNCA deben entrar como feature (IDs / leakage / auxiliares)
DROP_ALWAYS = [
    "store_id", "product_id",
    "week_start",
    "will_sell_next_7d", "revenue_next_7d",
    "units_sold_this_week"  # si target es venta próxima semana, esta se puede usar; aquí la dejamos fuera para ser conservadores
]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(ART_DIR, exist_ok=True)

    df = pd.read_parquet(RAW_PATH)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    # Target
    y = df[TARGET_CLASSIF].astype(int).to_numpy()

    # Features
    X = df.drop(columns=[c for c in DROP_ALWAYS if c in df.columns])

    # Split temporal 80/20
    cut = int(len(df) * 0.8)
    cutoff_date = df.loc[cut, TIME_COL]

    train_mask = df[TIME_COL] <= cutoff_date
    test_mask = df[TIME_COL] > cutoff_date

    X_train = X.loc[train_mask].copy()
    y_train = y[train_mask]

    X_test = X.loc[test_mask].copy()
    y_test = y[test_mask]

    # Detectar columnas
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # Pipelines (cloud-ready: soporta NaN y categorías nuevas)
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    # Fit SOLO en train (evita leakage)
    preprocess.fit(X_train)

    # Guardar datasets (sin transformar) + preprocess
    X_train.to_parquet(os.path.join(OUT_DIR, "X_train.parquet"), index=False)
    X_test.to_parquet(os.path.join(OUT_DIR, "X_test.parquet"), index=False)
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)

    dump({
        "preprocess": preprocess,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cutoff_date": str(cutoff_date),
        "target": TARGET_CLASSIF,
        "drop_always": DROP_ALWAYS,
    }, os.path.join(ART_DIR, "preprocess_module1.joblib"))

    # Validación rápida
    print("OK -> Preparación completada")
    print("RAW:", RAW_PATH)
    print("Cutoff:", cutoff_date)
    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Positive rate train:", y_train.mean().round(3), "test:", y_test.mean().round(3))
    print("Cat cols:", len(cat_cols), "Num cols:", len(num_cols))

if __name__ == "__main__":
    main()