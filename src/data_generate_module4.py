import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class GenConfig:
    seed: int = 42
    n_stores: int = 2000
    weeks: int = 16
    out_path: str = "data/raw/module4_content_raw.parquet"

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    cfg = GenConfig()
    np.random.seed(cfg.seed)
    os.makedirs(os.path.dirname(cfg.out_path), exist_ok=True)

    rows = []
    for store_id in range(1, cfg.n_stores + 1):
        base_quality = np.random.normal(0.0, 1.0)  # "calidad" latente del negocio
        base_audience = np.clip(np.random.lognormal(3.0, 0.6), 30, 40000)  # tamaño audiencia potencial

        business_type_slug = np.random.choice(["store", "appointments", "restaurant"], p=[0.6, 0.25, 0.15])
        has_instagram = int(np.random.binomial(1, 0.65))
        has_tiktok = int(np.random.binomial(1, 0.55))
        has_facebook = int(np.random.binomial(1, 0.70))
        has_whatsapp = int(np.random.binomial(1, 0.78))

        for week in range(1, cfg.weeks + 1):
            # actividad de contenido
            posts_count_30d = int(np.clip(np.random.poisson(8 + 2*max(base_quality, -1)), 0, 60))
            stories_count_30d = int(np.clip(np.random.poisson(14 + 3*max(base_quality, -1)), 0, 160))

            pct_posts_with_video = float(np.clip(np.random.normal(0.45, 0.25), 0.0, 1.0))
            pct_posts_carousel = float(np.clip(np.random.normal(0.22, 0.18), 0.0, 1.0))

            avg_views_per_post = float(np.clip(base_audience * np.random.uniform(0.02, 0.15) * (1 + 0.30*pct_posts_with_video), 20, 200000))
            avg_likes_per_post = float(np.clip(avg_views_per_post * np.random.uniform(0.01, 0.06), 0, 15000))
            avg_comments_per_post = float(np.clip(avg_views_per_post * np.random.uniform(0.0006, 0.006), 0, 1200))

            post_engagement_rate = float(np.clip((avg_likes_per_post + avg_comments_per_post) / (avg_views_per_post + 1e-9), 0.0001, 0.30))

            # reviews
            reviews_count = int(np.clip(np.random.poisson(8 + 2*max(base_quality, -1)), 0, 800))
            avg_rating = float(np.clip(np.random.normal(4.4 + 0.25*np.tanh(base_quality), 0.45), 1.0, 5.0))
            pct_1_2_star = float(np.clip(np.random.normal(0.05 - 0.02*np.tanh(base_quality), 0.04), 0.0, 0.35))
            reviews_with_photo_pct = float(np.clip(np.random.normal(0.18, 0.10), 0.0, 0.80))
            avg_review_length = float(np.clip(np.random.normal(65, 35), 5, 500))

            # frecuencia
            days_between_posts = float(np.clip(30 / (posts_count_30d + 1e-9), 0.3, 30.0))
            best_post_day_of_week = int(np.random.choice(range(7)))
            best_post_hour = int(np.random.choice(range(24), p=np.array([1]*24)/24))

            # demanda base actual
            profile_visits = float(np.clip(base_audience * np.random.uniform(0.03, 0.18), 10, 300000))
            story_view_rate = float(np.clip(np.random.normal(0.35, 0.15), 0.05, 0.95))
            avg_story_views = float(np.clip(profile_visits * story_view_rate * np.random.uniform(0.2, 1.1), 5, 250000))

            # target futuro (próxima semana)
            score = (
                + 1.2 * post_engagement_rate
                + 0.30 * np.log1p(posts_count_30d)
                + 0.15 * np.log1p(stories_count_30d)
                + 0.25 * pct_posts_with_video
                + 0.10 * pct_posts_carousel
                + 0.12 * (avg_rating - 4.0)
                - 0.90 * pct_1_2_star
                + 0.08 * reviews_with_photo_pct
                + 0.10 * (1 if has_instagram else 0)
                + 0.12 * (1 if has_tiktok else 0)
                + 0.10 * (1 if has_whatsapp else 0)
                - 0.06 * days_between_posts
                + 0.10 * np.tanh(base_quality)
            )

            p_growth = float(np.clip(sigmoid(score - 0.8), 0.03, 0.97))
            growth_next_week = int(np.random.binomial(1, p_growth))

            profile_visits_next_week = float(np.clip(profile_visits * (1 + np.random.normal(0.05, 0.10) + 0.25*p_growth), 5, 450000))
            new_customers_next_week = float(np.clip((profile_visits_next_week * np.random.uniform(0.002, 0.01)) * (1 + 0.35*p_growth), 0, 4500))

            rows.append({
                "store_id": store_id,
                "week": week,

                "business_type_slug": business_type_slug,
                "has_instagram": has_instagram,
                "has_tiktok": has_tiktok,
                "has_facebook": has_facebook,
                "has_whatsapp": has_whatsapp,

                "posts_count_30d": posts_count_30d,
                "stories_count_30d": stories_count_30d,
                "avg_likes_per_post": avg_likes_per_post,
                "avg_comments_per_post": avg_comments_per_post,
                "avg_views_per_post": avg_views_per_post,
                "post_engagement_rate": post_engagement_rate,
                "pct_posts_with_video": pct_posts_with_video,
                "pct_posts_carousel": pct_posts_carousel,
                "days_between_posts": days_between_posts,
                "best_post_day_of_week": best_post_day_of_week,
                "best_post_hour": best_post_hour,

                "reviews_count": reviews_count,
                "avg_rating": avg_rating,
                "pct_1_2_star": pct_1_2_star,
                "reviews_with_photo_pct": reviews_with_photo_pct,
                "avg_review_length": avg_review_length,

                "profile_visits": profile_visits,
                "avg_story_views": avg_story_views,
                "story_view_rate": story_view_rate,

                "growth_next_week": growth_next_week,
                "profile_visits_next_week": profile_visits_next_week,
                "new_customers_next_week": new_customers_next_week,
            })

    df = pd.DataFrame(rows)
    df.to_parquet(cfg.out_path, index=False)
    print(f"OK -> {cfg.out_path} | rows={len(df)} | growth_rate={df['growth_next_week'].mean():.3f}")

if __name__ == "__main__":
    main()