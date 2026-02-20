import os
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

RAW_PATH = os.getenv("TRIBIO_M4_RAW", "data/raw/module4_content_raw.parquet")
OUT_DIR = os.getenv("TRIBIO_M4_PROCESSED", "data/processed")
ART_DIR = os.getenv("TRIBIO_ARTIFACTS_DIR", "artifacts")

TARGET = "growth_next_week"
DROP_ALWAYS = ["store_id", "profile_visits_next_week", "new_customers_next_week", TARGET]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(ART_DIR, exist_ok=True)

    df = pd.read_parquet(RAW_PATH).reset_index(drop=True)

    # split temporal por semana (evita leakage)
    df_sorted = df.sort_values(["store_id", "week"]).reset_index(drop=True)
    cut = int(len(df_sorted) * 0.8)

    train_df = df_sorted.iloc[:cut].copy()
    test_df = df_sorted.iloc[cut:].copy()

    y_train = train_df[TARGET].astype(int).to_numpy()
    y_test = test_df[TARGET].astype(int).to_numpy()

    X_train = train_df.drop(columns=[c for c in DROP_ALWAYS if c in train_df.columns])
    X_test = test_df.drop(columns=[c for c in DROP_ALWAYS if c in test_df.columns])

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

    X_train.to_parquet(os.path.join(OUT_DIR, "m4_X_train.parquet"), index=False)
    X_test.to_parquet(os.path.join(OUT_DIR, "m4_X_test.parquet"), index=False)
    np.save(os.path.join(OUT_DIR, "m4_y_train.npy"), y_train)
    np.save(os.path.join(OUT_DIR, "m4_y_test.npy"), y_test)

    dump({
        "preprocess": preprocess,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "target": TARGET,
        "drop_always": DROP_ALWAYS,
    }, os.path.join(ART_DIR, "preprocess_module4.joblib"))

    print("OK -> Module4 preparado")
    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Growth rate train:", round(y_train.mean(), 3), "test:", round(y_test.mean(), 3))

if __name__ == "__main__":
    main()