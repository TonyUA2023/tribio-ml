import os
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

RAW_PATH = os.getenv("TRIBIO_M2_RAW", "data/raw/module2_customers_raw.parquet")
OUT_DIR = os.getenv("TRIBIO_M2_PROCESSED", "data/processed")
ART_DIR = os.getenv("TRIBIO_ARTIFACTS_DIR", "artifacts")
TARGET = "will_churn_30d"

DROP_ALWAYS = ["store_id", "customer_id", TARGET]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(ART_DIR, exist_ok=True)

    df = pd.read_parquet(RAW_PATH).reset_index(drop=True)

    y = df[TARGET].astype(int).to_numpy()
    X = df.drop(columns=[c for c in DROP_ALWAYS if c in df.columns])

    # split simple 80/20 (en real será temporal por ventanas; aquí es sintético por cliente)
    cut = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:cut].copy(), X.iloc[cut:].copy()
    y_train, y_test = y[:cut], y[cut:]

    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

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

    preprocess.fit(X_train)

    # guardar
    X_train.to_parquet(os.path.join(OUT_DIR, "m2_X_train.parquet"), index=False)
    X_test.to_parquet(os.path.join(OUT_DIR, "m2_X_test.parquet"), index=False)
    np.save(os.path.join(OUT_DIR, "m2_y_train.npy"), y_train)
    np.save(os.path.join(OUT_DIR, "m2_y_test.npy"), y_test)

    dump({
        "preprocess": preprocess,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "target": TARGET,
        "drop_always": DROP_ALWAYS,
    }, os.path.join(ART_DIR, "preprocess_module2.joblib"))

    print("OK -> Module2 preparado")
    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Churn rate train:", round(y_train.mean(), 3), "test:", round(y_test.mean(), 3))

if __name__ == "__main__":
    main()