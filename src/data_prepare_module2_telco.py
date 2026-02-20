import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_CSV = os.getenv("TRIBIO_M2_CSV", "data/raw/telco_churn.csv")
OUT_DIR = os.getenv("TRIBIO_M2_PROCESSED", "data/processed")

TARGET_COL = "Churn"  # Yes/No

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(RAW_CSV)

    # --- Limpieza
    # TotalCharges suele venir con espacios/strings vacíos
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")

    # SeniorCitizen a veces viene como 0/1 numérico -> ok
    # customerID no aporta al modelo
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Target binario
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0}).astype(int)

    # Separar X/y
    y = df[TARGET_COL].to_numpy().astype(int)
    X = df.drop(columns=[TARGET_COL])

    # --- Split estratificado (dataset no tiene fecha)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Guardar en formato que tu dashboard ya espera
    X_train.to_parquet(os.path.join(OUT_DIR, "m2_X_train.parquet"), index=False)
    X_test.to_parquet(os.path.join(OUT_DIR, "m2_X_test.parquet"), index=False)
    np.save(os.path.join(OUT_DIR, "m2_y_train.npy"), y_train)
    np.save(os.path.join(OUT_DIR, "m2_y_test.npy"), y_test)

    print("✅ M2 listo (Telco)")
    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Churn rate:", round(y.mean(), 3))

if __name__ == "__main__":
    main()