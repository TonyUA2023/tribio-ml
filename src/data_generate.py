# src/data_generate.py
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

@dataclass
class GenConfig:
    seed: int = 42
    n_stores: int = 120
    products_per_store: int = 80
    n_weeks: int = 52
    start_date: str = "2025-01-06"  # lunes
    out_path: str = "data/raw/module1_product_week_raw.parquet"

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    cfg = GenConfig()
    np.random.seed(cfg.seed)

    os.makedirs(os.path.dirname(cfg.out_path), exist_ok=True)

    # Catálogos
    business_types = ["store", "appointments", "restaurant"]
    categories = ["barber", "salon", "spa", "retail", "food", "education", "health", "agro"]
    plans = ["free", "basic", "pro", "enterprise"]
    devices = ["mobile", "desktop"]
    genders = ["male", "female", "unisex", "kids"]
    conditions = ["new", "used", "refurbished"]
    countries = ["PE", "BR", "CN", "US", "MX", "CO"]

    start = pd.to_datetime(cfg.start_date)
    weeks = [start + timedelta(days=7*i) for i in range(cfg.n_weeks)]

    rows = []

    for store_id in range(1, cfg.n_stores + 1):
        # Store-level attributes
        business_type = np.random.choice(business_types, p=[0.70, 0.15, 0.15])
        business_cat = np.random.choice(categories)
        plan_id = np.random.choice(plans, p=[0.45, 0.35, 0.15, 0.05])

        has_whatsapp = np.random.binomial(1, 0.78)
        has_instagram = np.random.binomial(1, 0.70)
        has_facebook = np.random.binomial(1, 0.55)
        has_tiktok = np.random.binomial(1, 0.35)
        payment_enabled = np.random.binomial(1, 0.55 if plan_id in ["pro", "enterprise"] else 0.35)

        store_age_days = int(np.random.gamma(shape=3.0, scale=120.0))  # promedio ~360 días
        total_products_active = int(max(10, min(250, np.random.normal(70, 25))))

        # Product catalog per store
        for pidx in range(1, cfg.products_per_store + 1):
            product_id = (store_id * 100000) + pidx

            # Product attributes (estables o semi-estables)
            base_price = float(np.random.lognormal(mean=3.7, sigma=0.45))  # ~40-80
            compare_price = base_price * np.random.choice([1.0, 1.0, 1.0, 1.10, 1.20, 1.30])  # a veces descuento
            price = base_price if compare_price == base_price else base_price * np.random.uniform(0.70, 0.95)

            has_discount = 1 if compare_price > price else 0
            discount_pct = float(((compare_price - price) / (compare_price + 1e-9)) * 100) if has_discount else 0.0

            # Stock: a veces ilimitado (None), a veces número
            stock_mode = np.random.choice(["finite", "infinite"], p=[0.80, 0.20])
            stock = None if stock_mode == "infinite" else int(np.random.poisson(35) + np.random.randint(0, 20))
            is_out_of_stock = 1 if (stock is not None and stock == 0) else 0

            featured = np.random.binomial(1, 0.12)
            available = np.random.binomial(1, 0.95)
            has_variants = np.random.binomial(1, 0.40)
            images_count = int(np.random.poisson(3) + 1)  # 1..8 aprox
            has_images_gallery = 1 if images_count > 1 else 0
            description_length = int(np.clip(np.random.normal(520, 260), 0, 2200))
            has_short_description = 1 if description_length > 80 else 0

            has_sku = np.random.binomial(1, 0.65)
            has_specifications = np.random.binomial(1, 0.40)
            specifications_count = int(np.random.poisson(4) if has_specifications else 0)

            gender = np.random.choice(genders, p=[0.25, 0.25, 0.40, 0.10])
            condition = np.random.choice(conditions, p=[0.85, 0.10, 0.05])
            origin_country = np.random.choice(countries, p=[0.55, 0.10, 0.20, 0.05, 0.05, 0.05])

            days_since_created = int(np.random.gamma(shape=2.5, scale=110.0))  # antigüedad producto
            category_depth = int(np.random.choice([0, 1, 2], p=[0.25, 0.55, 0.20]))

            has_brand = np.random.binomial(1, 0.55)
            brand_product_count = int(np.random.poisson(18) + 1) if has_brand else 0
            category_product_count = int(np.random.poisson(110) + 10)

            # Simulación temporal semana a semana (ventas, cancelaciones, pagos)
            # Creamos una "propensión" base de venta por producto, modulada por store + producto + marketing
            base_intent = (
                0.35
                + (0.08 if featured else 0.0)
                + (0.06 if has_images_gallery else -0.02)
                + (0.04 if description_length > 300 else -0.03)
                + (0.03 if has_specifications else 0.0)
                + (0.05 if plan_id in ["pro", "enterprise"] else 0.0)
                + (0.03 if payment_enabled else -0.01)
                + (0.02 if has_whatsapp else 0.0)
                + (0.02 if has_instagram else 0.0)
                - (0.04 if is_out_of_stock else 0.0)
            )

            # Precio “alto” reduce conversión ligeramente
            price_penalty = np.clip((price - 70) / 200, 0, 0.20)
            base_intent -= float(price_penalty)

            # Descuento ayuda, pero con saturación
            base_intent += float(np.clip(discount_pct / 100, 0, 0.15))

            # Historial (inicial) por ventanas
            sales_history = []

            for w, week_start in enumerate(weeks):
                # Estacionalidad simple
                month = int(pd.to_datetime(week_start).month)
                season_boost = 0.03 if month in [11, 12] else (0.02 if month in [2, 5] else 0.0)

                # “Tráfico” implícito: aumenta con redes/plan
                traffic = (
                    np.random.poisson(30)
                    + (10 if has_instagram else 0)
                    + (8 if has_tiktok else 0)
                    + (6 if has_facebook else 0)
                    + (12 if plan_id in ["pro", "enterprise"] else 0)
                )

                # Probabilidad de venta esa semana
                p_sell = sigmoid((base_intent + season_boost) * 2.2)  # ~0.30 a 0.80
                # Aumenta levemente con “tráfico”
                p_sell = float(np.clip(p_sell + (traffic / 500), 0.01, 0.95))

                will_sell = np.random.binomial(1, p_sell)

                # Unidades vendidas si vende
                units = int(np.random.poisson(2.0) + 1) if will_sell else 0

                # Ingresos
                revenue = float(units * price * np.random.uniform(0.95, 1.05))

                # Cancelación y éxito de pago (para señales de calidad)
                cancelled_rate = float(np.clip(np.random.normal(0.06, 0.04), 0.0, 0.30))
                payment_success_rate = float(np.clip(np.random.normal(0.88, 0.08), 0.50, 0.99))

                # Si el pago no está habilitado, baja success rate
                if not payment_enabled:
                    payment_success_rate = float(np.clip(payment_success_rate - 0.10, 0.40, 0.98))

                # Last sale days ago (aprox)
                # Si hubo venta esta semana, 0-6, si no, aumenta
                last_sale_days_ago = 0 if units > 0 else (sales_history[-1]["product_last_sale_days_ago"] + 7 if sales_history else 999)
                if units > 0:
                    last_sale_days_ago = int(np.random.randint(0, 7))

                # Ventanas agregadas con leakage control: ventas pasadas (no incluye semana futura)
                # Definimos sales_7d = units actual (como simplificación) y sales_30d = sum últimas 4 semanas previas
                past_4 = [h["product_sales_7d"] for h in sales_history[-4:]]
                product_sales_30d = int(np.sum(past_4))  # SOLO pasado
                product_sales_7d = int(units)            # semana actual (ok si target es FUTURA)
                # Importante: el target será FUTURO (semana siguiente), por eso units actual puede ser feature.

                # Guardamos fila
                rows.append({
                    "week_start": pd.to_datetime(week_start),
                    "store_id": store_id,
                    "product_id": product_id,

                    # Store features
                    "business_type_slug": business_type,
                    "business_category_slug": business_cat,
                    "plan_id": plan_id,
                    "has_whatsapp": has_whatsapp,
                    "has_instagram": has_instagram,
                    "has_facebook": has_facebook,
                    "has_tiktok": has_tiktok,
                    "payment_enabled": int(payment_enabled),
                    "store_age_days": store_age_days,
                    "total_products_active": total_products_active,

                    # Product features
                    "price": float(price),
                    "compare_price": float(compare_price),
                    "discount_pct": float(discount_pct),
                    "has_discount": int(has_discount),
                    "stock": -1 if stock is None else int(stock),  # -1 = ilimitado
                    "is_out_of_stock": int(is_out_of_stock),
                    "featured": int(featured),
                    "available": int(available),
                    "has_variants": int(has_variants),
                    "has_images_gallery": int(has_images_gallery),
                    "images_count": int(images_count),
                    "has_short_description": int(has_short_description),
                    "description_length": int(description_length),
                    "has_sku": int(has_sku),
                    "has_specifications": int(has_specifications),
                    "specifications_count": int(specifications_count),
                    "gender": gender,
                    "condition": condition,
                    "origin_country": origin_country,
                    "days_since_created": int(days_since_created),
                    "category_depth": int(category_depth),
                    "has_brand": int(has_brand),
                    "brand_product_count": int(brand_product_count),
                    "category_product_count": int(category_product_count),

                    # Historical/quality signals (PAST/Current only)
                    "product_sales_7d": int(product_sales_7d),
                    "product_sales_30d": int(product_sales_30d),
                    "product_revenue_7d": float(revenue),
                    "product_cancelled_rate": float(cancelled_rate),
                    "product_payment_success_rate": float(payment_success_rate),
                    "product_last_sale_days_ago": int(last_sale_days_ago),

                    # Temporal features
                    "day_of_week": int(pd.to_datetime(week_start).dayofweek),
                    "month": int(month),
                    "is_weekend": int(pd.to_datetime(week_start).dayofweek >= 5),

                    # Target placeholder (se crea después con shift)
                    "units_sold_this_week": int(units),
                })

                sales_history.append({
                    "product_sales_7d": int(units),
                    "product_last_sale_days_ago": int(last_sale_days_ago),
                })

    df = pd.DataFrame(rows)

    # Crear targets FUTUROS (sin leakage):
    # will_sell_next_7d = units_sold en semana siguiente > 0
    df = df.sort_values(["store_id", "product_id", "week_start"]).reset_index(drop=True)
    df["units_next_week"] = df.groupby(["store_id", "product_id"])["units_sold_this_week"].shift(-1)
    df["will_sell_next_7d"] = (df["units_next_week"].fillna(0) > 0).astype(int)
    df["revenue_next_7d"] = df.groupby(["store_id", "product_id"])["product_revenue_7d"].shift(-1).fillna(0.0)

    # Limpiar columnas auxiliares
    df = df.drop(columns=["units_next_week"])

    # Guardar
    df.to_parquet(cfg.out_path, index=False)
    print(f"OK -> {cfg.out_path} | rows={len(df)} | positive_rate={df['will_sell_next_7d'].mean():.3f}")

if __name__ == "__main__":
    main()