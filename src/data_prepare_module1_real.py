import os
import numpy as np
import pandas as pd

RAW_CSV = os.getenv("TRIBIO_M1_CSV", "data/raw/online_retail.csv")
OUT_RAW = "data/raw/module1_product_day_raw.parquet"

OUT_DIR = "data/processed"
TARGET = "will_sell_next_7d"

def build_forward_sum(series: pd.Series, window: int) -> pd.Series:
    # suma futura de los próximos "window" días (t+1..t+window)
    rev = series.iloc[::-1]
    fut = rev.rolling(window=window, min_periods=1).sum().shift(1)
    return fut.iloc[::-1]

def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(RAW_CSV)

    # Parse datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate", "StockCode", "InvoiceNo"])

    # Limpieza básica
    df["Description"] = df["Description"].fillna("")
    df["CustomerID"] = df["CustomerID"].fillna(-1).astype(float).astype(int)
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df = df.dropna(subset=["UnitPrice", "Quantity"])

    # Cancelaciones / devoluciones
    df["is_cancel"] = df["InvoiceNo"].astype(str).str.startswith("C") | (df["Quantity"] < 0)

    # Totales
    df["line_total"] = df["Quantity"] * df["UnitPrice"]

    # Nos quedamos con fechas a nivel día
    df["date"] = df["InvoiceDate"].dt.floor("D")

    # Agregación diaria por producto
    g = df.groupby(["StockCode", "date"], as_index=False)

    daily = g.agg(
        price=("UnitPrice", "mean"),
        description_length=("Description", lambda s: int(np.mean([len(x) for x in s.astype(str).values])) if len(s) else 0),
        sold_qty=("Quantity", lambda s: float(np.sum(s[s > 0]))),
        revenue=("line_total", lambda s: float(np.sum(s[s > 0]))),
        cancelled_qty=("Quantity", lambda s: float(np.sum(np.abs(s[s < 0])))),
        cancelled_rows=("is_cancel", "sum"),
        rows=("InvoiceNo", "count"),
        unique_buyers=("CustomerID", lambda s: int(pd.Series(s).nunique())),
    )

    daily = daily.sort_values(["StockCode", "date"]).reset_index(drop=True)

    # ROLLINGS / VENTANAS
    def add_rollings(pdf: pd.DataFrame) -> pd.DataFrame:
        pdf = pdf.sort_values("date").reset_index(drop=True)

        pdf["product_sales_7d"] = pdf["sold_qty"].rolling(7, min_periods=1).sum()
        pdf["product_sales_30d"] = pdf["sold_qty"].rolling(30, min_periods=1).sum()
        pdf["product_revenue_7d"] = pdf["revenue"].rolling(7, min_periods=1).sum()
        pdf["product_revenue_30d"] = pdf["revenue"].rolling(30, min_periods=1).sum()

        # compradores únicos 30d (aprox): rolling de conteo no es exacto por duplicados, pero sirve como proxy
        pdf["product_unique_buyers_30d"] = pdf["unique_buyers"].rolling(30, min_periods=1).mean()

        # tasa de cancelación (proxy)
        canc = pdf["cancelled_rows"].rolling(30, min_periods=1).sum()
        tot = pdf["rows"].rolling(30, min_periods=1).sum()
        pdf["product_cancelled_rate_30d"] = (canc / (tot + 1e-9)).clip(0, 1)

        # días desde última venta
        last_sale = pdf["date"].where(pdf["sold_qty"] > 0).ffill()
        pdf["product_last_sale_days_ago"] = (pdf["date"] - last_sale).dt.days.fillna(999).clip(0, 999)

        # TARGET FUTURO (próximos 7 días)
        future_sales_7d = build_forward_sum(pdf["sold_qty"], window=7)
        pdf[TARGET] = (future_sales_7d > 0).astype(int)

        return pdf

    daily = daily.groupby("StockCode", group_keys=False).apply(add_rollings)

    # Features temporales
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["week_of_year"] = daily["date"].dt.isocalendar().week.astype(int)
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)

    # Campos que no existen en el dataset (los ponemos fijos para compatibilidad / baseline)
    daily["featured"] = 0
    daily["images_count"] = 1
    daily["has_specifications"] = 0
    daily["discount_pct"] = 0.0
    daily["stock"] = -1
    daily["plan_id"] = "basic"
    daily["business_type_slug"] = "store"
    daily["business_category_slug"] = "retail"
    daily["payment_settings_enabled"] = 1
    daily["has_whatsapp"] = 1

    # Filtrado para evitar “leakage” y filas sin historia suficiente
    # (pedimos mínimo 30 días de historial para features estables)
    daily = daily.sort_values(["StockCode", "date"]).reset_index(drop=True)
    daily = daily[daily["product_sales_30d"].notna()]

    # Quitamos últimos 7 días (no hay futuro completo)
    max_date = daily["date"].max()
    daily = daily[daily["date"] <= (max_date - pd.Timedelta(days=7))].copy()

    daily.to_parquet(OUT_RAW, index=False)
    print(f"OK -> {OUT_RAW} | rows={len(daily)} | target_rate={daily[TARGET].mean():.3f}")

    # ============================
    # Crear X_train/X_test + y
    # Split temporal (80/20)
    # ============================
    daily = daily.sort_values("date")
    cut = int(len(daily) * 0.8)
    cutoff_date = daily["date"].iloc[cut]

    train_df = daily[daily["date"] <= cutoff_date].copy()
    test_df = daily[daily["date"] > cutoff_date].copy()

    cols_drop = ["StockCode", "date", "sold_qty", "revenue", "cancelled_qty", "cancelled_rows", "rows", "unique_buyers", TARGET]
    feature_cols = [c for c in daily.columns if c not in cols_drop]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET].to_numpy().astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df[TARGET].to_numpy().astype(int)

    X_train.to_parquet(f"{OUT_DIR}/X_train.parquet", index=False)
    X_test.to_parquet(f"{OUT_DIR}/X_test.parquet", index=False)
    np.save(f"{OUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUT_DIR}/y_test.npy", y_test)

    print("OK -> data/processed/X_train.parquet, X_test.parquet, y_train.npy, y_test.npy")
    print("Train:", X_train.shape, "Test:", X_test.shape)

if __name__ == "__main__":
    main()