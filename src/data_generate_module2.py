import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class GenConfig:
    seed: int = 42
    n_stores: int = 120
    customers_per_store: int = 600
    out_path: str = "data/raw/module2_customers_raw.parquet"

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    cfg = GenConfig()
    np.random.seed(cfg.seed)
    os.makedirs(os.path.dirname(cfg.out_path), exist_ok=True)

    pay_methods = ["cash", "card", "transfer"]
    notif = ["email", "whatsapp", "sms"]

    rows = []
    for store_id in range(1, cfg.n_stores + 1):
        for cidx in range(1, cfg.customers_per_store + 1):
            customer_id = store_id * 100000 + cidx

            # RFM base
            days_since_last_order = int(np.clip(np.random.gamma(2.0, 12.0), 0, 180))
            total_orders_paid = int(np.clip(np.random.poisson(2.5), 0, 60))
            avg_order_value = float(np.clip(np.random.lognormal(3.6, 0.35), 10, 800))
            total_spent = float(total_orders_paid * avg_order_value * np.random.uniform(0.85, 1.15))

            # comportamiento
            cancellation_rate = float(np.clip(np.random.normal(0.06, 0.05), 0.0, 0.40))
            orders_weekend_pct = float(np.clip(np.random.normal(0.25, 0.18), 0.0, 1.0))
            avg_qty_per_order = float(np.clip(np.random.normal(1.8, 0.9), 1.0, 10.0))

            # engagement
            profile_visits_count = int(np.clip(np.random.poisson(8) + (total_orders_paid * 2), 0, 500))
            link_clicks_count = int(np.clip(np.random.poisson(4) + (profile_visits_count * 0.25), 0, 800))
            link_click_to_order_ratio = float(link_clicks_count / (total_orders_paid + 1))

            has_written_review = int(np.random.binomial(1, 0.18 + 0.01 * min(total_orders_paid, 10)))
            avg_review_rating = float(np.clip(np.random.normal(4.4, 0.5), 1.0, 5.0)) if has_written_review else np.nan

            preferred_payment_method = np.random.choice(pay_methods, p=[0.45, 0.40, 0.15])
            preferred_notification = np.random.choice(notif, p=[0.20, 0.65, 0.15])

            # Target churn (más recency, menos frequency/monetary, más cancelación => más churn)
            score = (
                + 0.030 * days_since_last_order
                - 0.080 * total_orders_paid
                - 0.0008 * total_spent
                + 2.200 * cancellation_rate
                - 0.0030 * profile_visits_count
                + 0.120 * link_click_to_order_ratio
            )
            p_churn = float(np.clip(sigmoid(score - 2.0), 0.01, 0.95))
            will_churn_30d = int(np.random.binomial(1, p_churn))

            rows.append({
                "store_id": store_id,
                "customer_id": customer_id,

                "is_registered": int(np.random.binomial(1, 0.45)),
                "days_since_last_order": days_since_last_order,
                "total_orders_paid": total_orders_paid,
                "avg_order_value": avg_order_value,
                "total_spent": total_spent,

                "cancellation_rate": cancellation_rate,
                "orders_weekend_pct": orders_weekend_pct,
                "avg_qty_per_order": avg_qty_per_order,

                "profile_visits_count": profile_visits_count,
                "link_click_to_order_ratio": link_click_to_order_ratio,
                "has_written_review": has_written_review,
                "avg_review_rating": avg_review_rating,

                "preferred_payment_method": preferred_payment_method,
                "preferred_notification": preferred_notification,

                "will_churn_30d": will_churn_30d
            })

    df = pd.DataFrame(rows)
    df.to_parquet(cfg.out_path, index=False)
    print(f"OK -> {cfg.out_path} | rows={len(df)} | churn_rate={df['will_churn_30d'].mean():.3f}")

if __name__ == "__main__":
    main()