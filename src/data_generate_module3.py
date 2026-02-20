import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class GenConfig:
    seed: int = 42
    n_stores: int = 2500
    out_path: str = "data/raw/module3_store_design_raw.parquet"

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    cfg = GenConfig()
    np.random.seed(cfg.seed)
    os.makedirs(os.path.dirname(cfg.out_path), exist_ok=True)

    business_types = ["store", "appointments", "restaurant"]
    templates = ["NikeStyle", "Valentine", "Academy", "Minimal", "DarkLuxury", "Classic"]

    rows = []
    for store_id in range(1, cfg.n_stores + 1):
        business_type_slug = np.random.choice(business_types, p=[0.60, 0.25, 0.15])
        template_slug = np.random.choice(templates, p=[0.20, 0.12, 0.12, 0.26, 0.10, 0.20])

        store_age_days = int(np.clip(np.random.gamma(2.0, 80.0), 7, 1800))
        plan_id = np.random.choice(["free", "basic", "pro", "enterprise"], p=[0.45, 0.35, 0.17, 0.03])

        # Config visual
        has_topbar = int(np.random.binomial(1, 0.55))
        topbar_has_text = int(np.random.binomial(1, 0.75 if has_topbar else 0.05))
        hero_slides_count = int(np.clip(np.random.poisson(2.0), 0, 6))
        hero_has_cta = int(np.random.binomial(1, 0.70 if hero_slides_count > 0 else 0.10))
        navigation_menu_items_count = int(np.clip(np.random.normal(5, 2), 0, 12))
        has_promo_banners = int(np.random.binomial(1, 0.45))
        promo_banners_count = int(np.clip(np.random.poisson(1.2) if has_promo_banners else 0, 0, 6))
        features_section_enabled = int(np.random.binomial(1, 0.55))
        features_count = int(np.clip(np.random.poisson(4.0) if features_section_enabled else 0, 0, 12))
        footer_has_links = int(np.random.binomial(1, 0.60))

        has_custom_logo = int(np.random.binomial(1, 0.70))
        has_cover_image = int(np.random.binomial(1, 0.62))
        media_count = int(np.clip(np.random.poisson(6.0), 0, 60))

        # color (representación simple)
        color_hue = float(np.clip(np.random.uniform(0, 360), 0, 360))
        color_saturation = float(np.clip(np.random.normal(0.65, 0.18), 0.05, 1.0))
        color_lightness = float(np.clip(np.random.normal(0.45, 0.15), 0.05, 0.95))

        # Catálogo
        total_products_active = int(np.clip(np.random.normal(60, 35), 5, 400))
        products_with_image_pct = float(np.clip(np.random.normal(0.78, 0.18), 0.05, 1.0))
        products_with_description_pct = float(np.clip(np.random.normal(0.62, 0.20), 0.05, 1.0))
        products_with_discount_pct = float(np.clip(np.random.normal(0.28, 0.18), 0.0, 1.0))
        products_featured_pct = float(np.clip(np.random.normal(0.12, 0.10), 0.0, 0.8))

        avg_product_price = float(np.clip(np.random.lognormal(3.7, 0.4), 10, 1500))
        price_range = float(np.clip(avg_product_price * np.random.uniform(0.4, 2.5), 5, 3000))

        categories_count = int(np.clip(np.random.normal(8, 5), 1, 60))
        brands_count = int(np.clip(np.random.normal(4, 3), 0, 40))
        avg_images_per_product = float(np.clip(np.random.normal(2.6, 1.2), 0.2, 12))

        payment_settings_enabled = int(np.random.binomial(1, 0.30 if plan_id == "free" else 0.68 if plan_id == "basic" else 0.85))
        has_whatsapp = int(np.random.binomial(1, 0.75))
        has_instagram = int(np.random.binomial(1, 0.55))

        # Target: alta conversión (clasificación) y conversion_rate (regresión)
        # score: catálogo completo + CTA + pagos + calidad de contenido suben conversión
        score = (
            + 0.9 * payment_settings_enabled
            + 0.6 * hero_has_cta
            + 0.25 * (hero_slides_count > 0)
            + 0.20 * (features_count >= 3)
            + 0.35 * has_custom_logo
            + 0.30 * has_cover_image
            + 0.45 * products_with_image_pct
            + 0.35 * products_with_description_pct
            + 0.25 * products_with_discount_pct
            + 0.10 * products_featured_pct
            + 0.10 * np.log1p(total_products_active)
            - 0.10 * (avg_product_price > 800)  # muy caro sin confianza suele bajar
            - 0.08 * (navigation_menu_items_count > 9)  # menú muy largo confunde
            + (0.25 if plan_id in ["pro", "enterprise"] else 0.0)
            + (0.08 if template_slug in ["NikeStyle", "Minimal", "DarkLuxury"] else 0.0)
        )

        p_high = float(np.clip(sigmoid(score - 1.4), 0.02, 0.98))
        high_conversion_30d = int(np.random.binomial(1, p_high))

        # conversion_rate aproximado (0–0.20)
        base_cr = 0.02 + 0.12 * p_high
        noise = np.random.normal(0, 0.01)
        conversion_rate_30d = float(np.clip(base_cr + noise, 0.001, 0.25))

        rows.append({
            "store_id": store_id,
            "business_type_slug": business_type_slug,
            "plan_id": plan_id,
            "template_slug": template_slug,
            "store_age_days": store_age_days,

            "has_topbar": has_topbar,
            "topbar_has_text": topbar_has_text,
            "hero_slides_count": hero_slides_count,
            "hero_has_cta": hero_has_cta,
            "navigation_menu_items_count": navigation_menu_items_count,
            "has_promo_banners": has_promo_banners,
            "promo_banners_count": promo_banners_count,
            "features_section_enabled": features_section_enabled,
            "features_count": features_count,
            "footer_has_links": footer_has_links,
            "has_custom_logo": has_custom_logo,
            "has_cover_image": has_cover_image,
            "media_count": media_count,

            "color_hue": color_hue,
            "color_saturation": color_saturation,
            "color_lightness": color_lightness,

            "total_products_active": total_products_active,
            "products_with_image_pct": products_with_image_pct,
            "products_with_description_pct": products_with_description_pct,
            "products_with_discount_pct": products_with_discount_pct,
            "products_featured_pct": products_featured_pct,
            "avg_product_price": avg_product_price,
            "price_range": price_range,
            "categories_count": categories_count,
            "brands_count": brands_count,
            "avg_images_per_product": avg_images_per_product,

            "payment_settings_enabled": payment_settings_enabled,
            "has_whatsapp": has_whatsapp,
            "has_instagram": has_instagram,

            "high_conversion_30d": high_conversion_30d,
            "conversion_rate_30d": conversion_rate_30d,
        })

    df = pd.DataFrame(rows)
    df.to_parquet(cfg.out_path, index=False)
    print(f"OK -> {cfg.out_path} | rows={len(df)} | high_rate={df['high_conversion_30d'].mean():.3f} | cr_mean={df['conversion_rate_30d'].mean():.3f}")

if __name__ == "__main__":
    main()