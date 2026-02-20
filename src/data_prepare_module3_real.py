import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_EVENTS = "data/raw/events1.csv"
RAW_ITEMS  = "data/raw/items.csv"
RAW_USERS  = "data/raw/users.csv"

OUT_DIR = "data/processed"
TARGET = "high_conversion_session"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    events = pd.read_csv(RAW_EVENTS)

    # Parse date
    events["date"] = pd.to_datetime(events["date"], errors="coerce")

    # Crear flags de eventos
    events["is_purchase"] = (events["type"] == "purchase").astype(int)
    events["is_pageview"] = (events["type"] == "page_view").astype(int)
    events["is_add_to_cart"] = (events["type"] == "add_to_cart").astype(int)

    # Agrupar por sesión
    g = events.groupby("ga_session_id")

    session_df = g.agg(
        user_id=("user_id", "first"),
        country=("country", "first"),
        device=("device", "first"),
        events_count=("type", "count"),
        pageviews_count=("is_pageview", "sum"),
        add_to_cart_count=("is_add_to_cart", "sum"),
        purchases=("is_purchase", "sum"),
        unique_items_viewed=("item_id", "nunique"),
        session_date=("date", "min")
    ).reset_index()

    # Target
    session_df[TARGET] = (session_df["purchases"] > 0).astype(int)

    # Features temporales
    session_df["session_day_of_week"] = session_df["session_date"].dt.dayofweek
    session_df["session_month"] = session_df["session_date"].dt.month

    # UX features derivadas
    session_df["is_mobile"] = session_df["device"].str.contains("mobile", case=False, na=False).astype(int)
    session_df["bounce"] = (session_df["pageviews_count"] <= 1).astype(int)

    # Drop columnas no necesarias
    session_df = session_df.drop(columns=["purchases", "session_date", "ga_session_id", "user_id"])

    # Separar X / y
    y = session_df[TARGET].to_numpy().astype(int)
    X = session_df.drop(columns=[TARGET])

    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Guardar
    X_train.to_parquet(f"{OUT_DIR}/m3_X_train.parquet", index=False)
    X_test.to_parquet(f"{OUT_DIR}/m3_X_test.parquet", index=False)
    np.save(f"{OUT_DIR}/m3_y_train.npy", y_train)
    np.save(f"{OUT_DIR}/m3_y_test.npy", y_test)

    print("✅ M3 listo (Google Merchandise)")
    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Conversion rate:", round(y.mean(), 3))

if __name__ == "__main__":
    main()
    