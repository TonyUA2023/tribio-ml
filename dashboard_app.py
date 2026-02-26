import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from joblib import load

st.set_page_config(page_title="TRIBIO IA - Dashboard", layout="wide")

ART_DIR = "artifacts"
PROCESSED_DIR = "data/processed"


# =========================
# Loaders (cache-friendly)
# =========================
@st.cache_resource
def try_load(path: str):
    if os.path.exists(path):
        return load(path)
    return None

@st.cache_resource
def load_models():
    m1 = try_load(f"{ART_DIR}/model_module1.joblib")
    m2 = try_load(f"{ART_DIR}/model_module2.joblib")
    m3 = try_load(f"{ART_DIR}/model_module3.joblib")
    m4 = try_load(f"{ART_DIR}/model_module4.joblib")
    return m1, m2, m3, m4

@st.cache_data
def load_xy(x_path: str, y_path: str):
    X = pd.read_parquet(x_path)
    y = np.load(y_path)
    return X, y


# =========================
# Helper: Confusion matrix df
# =========================
def confusion_df(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    return pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Real 0", "Real 1"],
        columns=["Pred 0", "Pred 1"]
    )


# =========================
# App
# =========================
st.title("📊 TRIBIO IA — Dashboard")
st.caption("M1: Ventas | M2: Clientes | M3: Diseño/Plantilla | M4: Contenido/Engagement")

m1, m2, m3, m4 = load_models()

missing = []
if m1 is None: missing.append("model_module1.joblib")
if m2 is None: missing.append("model_module2.joblib")
if m3 is None: missing.append("model_module3.joblib")
if m4 is None: missing.append("model_module4.joblib")

if missing:
    st.warning("Faltan modelos en /artifacts: " + ", ".join(missing) +
               ". Se mostrarán solo los módulos disponibles.")

tabs = []
tab_labels = []
if m1 is not None:
    tab_labels.append("🟦 Módulo 1: Ventas")
if m2 is not None:
    tab_labels.append("🟩 Módulo 2: Clientes (Churn)")
if m3 is not None:
    tab_labels.append("🟪 Módulo 3: Diseño/Config")
if m4 is not None:
    tab_labels.append("🟧 Módulo 4: Contenido/Engagement")

if not tab_labels:
    st.error("No hay ningún modelo cargado. Entrena al menos el Módulo 1 y/o 2 para iniciar.")
    st.stop()

tabs = st.tabs(tab_labels)

tab_idx = 0


# =========================
# TAB 1 — Module 1 Sales
# =========================
if m1 is not None:
    with tabs[tab_idx]:
        st.header("🟦 Módulo 1 — Predicción de Ventas + Recomendaciones")

        pipe = m1["pipeline"]
        thr = float(m1["threshold"])
        model_name = m1["model_name"]
        target = m1.get("target", "will_sell_next_7d")

        # Datos
        try:
            X_test, y_test = load_xy(
                f"{PROCESSED_DIR}/X_test.parquet",
                f"{PROCESSED_DIR}/y_test.npy"
            )
        except Exception as e:
            st.error("No se pudieron cargar los datos del módulo 1. Verifica data/processed/X_test.parquet y y_test.npy")
            st.exception(e)
            st.stop()

        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= thr).astype(int)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Modelo", model_name)
        col2.metric("Threshold", f"{thr:.3f}")
        col3.metric("Prob. promedio", f"{proba.mean():.3f}")
        col4.metric("Tasa predicha (1)", f"{pred.mean():.3f}")

        st.divider()

        st.subheader("📈 Distribución de Probabilidades")
        st.plotly_chart(px.histogram(proba, nbins=40, title="Probabilidades de venta próxima semana"),
                        use_container_width=True)

        st.subheader("🧩 Matriz de confusión (threshold actual)")
        st.dataframe(confusion_df(y_test, pred), use_container_width=True)

        st.divider()

        st.subheader("🔥 Top predicciones (productos con mayor probabilidad)")
        top_n = st.slider("Top N", 5, 100, 20, key="m1_topn")

        table = X_test.copy()
        table["proba_sell_next_7d"] = proba
        table["pred"] = pred

        with st.expander("Filtros"):
            if "business_category_slug" in table.columns:
                cats = sorted(table["business_category_slug"].astype(str).unique())
                sel_cat = st.multiselect("Categoría negocio", cats,
                                         default=cats[:3] if len(cats) >= 3 else cats, key="m1_cat")
                table = table[table["business_category_slug"].astype(str).isin(sel_cat)]

            if "plan_id" in table.columns:
                plans = sorted(table["plan_id"].astype(str).unique())
                sel_plan = st.multiselect("Plan", plans, default=plans, key="m1_plan")
                table = table[table["plan_id"].astype(str).isin(sel_plan)]

        st.dataframe(table.sort_values("proba_sell_next_7d", ascending=False).head(top_n),
                     use_container_width=True)

        st.divider()

        st.subheader("🧠 Probar un nuevo producto (simulación)")
        with st.form("nuevo_producto"):
            c1, c2, c3, c4 = st.columns(4)
            price = c1.number_input("price", min_value=1.0, value=55.0)
            discount_pct = c2.number_input("discount_pct", min_value=0.0, max_value=90.0, value=10.0)
            stock = c3.number_input("stock (-1 ilimitado)", value=30)
            images_count = c4.number_input("images_count", min_value=0, value=3)

            c5, c6, c7, c8 = st.columns(4)
            description_length = c5.number_input("description_length (chars)", min_value=0, value=300, help="Caracteres en la descripción del producto (ej: 300 = descripción media)")
            featured = c6.selectbox("featured", [0, 1], index=0)
            payment_settings_enabled = c7.selectbox("payment_settings_enabled", [0, 1], index=1)
            has_whatsapp = c8.selectbox("has_whatsapp", [0, 1], index=1)

            c9, c10, c11 = st.columns(3)
            plan_id_in = c9.selectbox("plan_id", ["free", "basic", "pro", "enterprise"], index=1)
            business_type_in = c10.selectbox("business_type_slug", ["store", "appointments", "restaurant"], index=0)
            business_cat_in = c11.selectbox("business_category_slug",
                                            ["retail", "barber", "salon", "spa", "food", "education", "health", "agro"],
                                            index=0)

            submit = st.form_submit_button("Predecir")

        if submit:
            # Scoring basado en reglas (el modelo M1 fue entrenado con
            # description_length como longitud media de nombres de productos
            # wholesale 0-35 chars, incompatible con descripciones Tribio)
            score = 0.0
            if 5 <= price <= 200:
                score += 0.20
            elif price < 5 or price > 500:
                score += 0.05
            else:
                score += 0.12

            if discount_pct >= 10:
                score += 0.15
            elif discount_pct > 0:
                score += 0.08

            if stock == 0:
                score += 0.00
            elif stock == -1 or stock >= 10:
                score += 0.15
            else:
                score += 0.08

            if images_count >= 4:
                score += 0.15
            elif images_count >= 2:
                score += 0.10
            elif images_count == 1:
                score += 0.04

            if description_length >= 400:
                score += 0.10
            elif description_length >= 150:
                score += 0.06
            elif description_length >= 50:
                score += 0.02

            if payment_settings_enabled == 1:
                score += 0.10
            if has_whatsapp == 1:
                score += 0.05
            if featured == 1:
                score += 0.05

            plan_bonus = {"free": 0.0, "basic": 0.03, "pro": 0.05, "enterprise": 0.05}
            score += plan_bonus.get(plan_id_in, 0.02)

            p = round(min(score, 1.0), 4)
            thr_sim = 0.55
            yhat = int(p >= thr_sim)

            st.success(f"Probabilidad de venta próxima semana: {p:.3f}")
            st.write("Predicción:", "SI vende ✅" if yhat else "NO vende ❌")

            rec = []
            if discount_pct == 0:
                rec.append("Prueba un descuento suave (5%–10%) o envío gratis para empujar la primera compra.")
            if images_count < 3:
                rec.append("Mejora la ficha: agrega 3–5 fotos reales (frontal, detalle, uso).")
            if description_length < 150:
                rec.append("Mejora la descripción: beneficios, compatibilidad, garantía, FAQs (mínimo 150 chars).")
            if stock == 0:
                rec.append("Stock en 0: reabastece o muestra alternativas para no cortar el embudo.")
            if payment_settings_enabled == 0:
                rec.append("Activa pagos online: reduce fricción y mejora conversión.")
            if not rec:
                rec.append("Ficha sólida: prueba A/B de precio y CTA para subir ticket y conversión.")

            st.subheader("✅ Recomendaciones")
            for r in rec:
                st.write("•", r)

    tab_idx += 1


# =========================
# TAB 2 — Module 2 Churn
# =========================
if m2 is not None:
    with tabs[tab_idx]:
        st.header("🟩 Módulo 2 — Clientes: Predicción de Churn + Recomendaciones")

        pipe = m2["pipeline"]
        thr = float(m2["threshold"])
        model_name = m2["model_name"]

        try:
            X_test, y_test = load_xy(
                f"{PROCESSED_DIR}/m2_X_test.parquet",
                f"{PROCESSED_DIR}/m2_y_test.npy"
            )
        except Exception as e:
            st.error("No se pudieron cargar los datos del módulo 2. Verifica data/processed/m2_X_test.parquet y m2_y_test.npy")
            st.exception(e)
            st.stop()

        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= thr).astype(int)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Modelo", model_name)
        c2.metric("Threshold", f"{thr:.3f}")
        c3.metric("Riesgo promedio churn", f"{proba.mean():.3f}")
        c4.metric("% clientes en riesgo", f"{pred.mean():.3f}")

        st.divider()

        st.subheader("📈 Distribución de riesgo de churn")
        st.plotly_chart(px.histogram(proba, nbins=40, title="Probabilidades de churn (30 días)"),
                        use_container_width=True)

        st.subheader("🧩 Matriz de confusión (threshold actual)")
        st.dataframe(confusion_df(y_test, pred), use_container_width=True)

        st.divider()

        st.subheader("🔥 Top clientes con mayor riesgo")
        top_n2 = st.slider("Top N clientes en riesgo", 10, 300, 50, key="m2_topn")

        tbl = X_test.copy()
        tbl["proba_churn_30d"] = proba
        tbl["pred_churn"] = pred
        st.dataframe(tbl.sort_values("proba_churn_30d", ascending=False).head(top_n2),
                     use_container_width=True)

        st.divider()

        st.subheader("🧠 Simular un cliente y obtener recomendaciones")
        with st.form("nuevo_cliente"):
            col1, col2, col3, col4 = st.columns(4)
            days_since_last_order = col1.number_input("days_since_last_order", min_value=0, value=25)
            total_orders_paid = col2.number_input("total_orders_paid", min_value=0, value=2)
            avg_order_value = col3.number_input("avg_order_value", min_value=1.0, value=70.0)
            cancellation_rate = col4.number_input("cancellation_rate", min_value=0.0, max_value=1.0, value=0.05)

            col5, col6, col7, col8 = st.columns(4)
            profile_visits_count = col5.number_input("profile_visits_count", min_value=0, value=10)
            link_click_to_order_ratio = col6.number_input("link_click_to_order_ratio", min_value=0.0, value=1.2)
            preferred_payment_method = col7.selectbox("preferred_payment_method", ["cash", "card", "transfer"], index=0)
            preferred_notification = col8.selectbox("preferred_notification", ["email", "whatsapp", "sms"], index=1)

            submit2 = st.form_submit_button("Evaluar churn")

        if submit2:
            new = {c: None for c in X_test.columns}

            if "days_since_last_order" in new: new["days_since_last_order"] = days_since_last_order
            if "total_orders_paid" in new: new["total_orders_paid"] = total_orders_paid
            if "avg_order_value" in new: new["avg_order_value"] = avg_order_value
            if "total_spent" in new: new["total_spent"] = float(total_orders_paid * avg_order_value)
            if "cancellation_rate" in new: new["cancellation_rate"] = cancellation_rate
            if "profile_visits_count" in new: new["profile_visits_count"] = profile_visits_count
            if "link_click_to_order_ratio" in new: new["link_click_to_order_ratio"] = link_click_to_order_ratio

            if "preferred_payment_method" in new: new["preferred_payment_method"] = preferred_payment_method
            if "preferred_notification" in new: new["preferred_notification"] = preferred_notification
            if "is_registered" in new: new["is_registered"] = 1

            new_df = pd.DataFrame([new])
            p = float(pipe.predict_proba(new_df)[:, 1][0])
            yhat = int(p >= thr)

            st.success(f"Riesgo de churn (30 días): {p:.3f}")
            st.write("Predicción:", "EN RIESGO ⚠️" if yhat else "ESTABLE ✅")

            rec = []
            if days_since_last_order >= 30:
                rec.append("Enviar WhatsApp personalizado con oferta limitada (alta recencia).")
            if total_orders_paid <= 1:
                rec.append("Activar campaña de bienvenida / primera recompra (cliente nuevo).")
            if cancellation_rate >= 0.12:
                rec.append("Reducir fricciones: claridad de precios, tiempos de entrega, métodos de pago.")
            if profile_visits_count >= 15 and total_orders_paid == 0:
                rec.append("Alta intención sin compra: cupón 5–10% + asistencia rápida por WhatsApp.")
            if preferred_payment_method == "cash":
                rec.append("Ofrecer transferencia/tarjeta con incentivo (reduce abandono).")
            if not rec:
                rec.append("Cliente saludable: upsell suave o programa de fidelidad para aumentar LTV.")

            st.subheader("✅ Recomendaciones")
            for r in rec:
                st.write("•", r)

    tab_idx += 1


# =========================
# TAB 3 — Module 3 Design/Config
# =========================
if m3 is not None:
    with tabs[tab_idx]:
        st.header("🟪 Módulo 3 — Diseño/Configuración: Impacto en Conversión + Recomendaciones")

        pipe = m3["pipeline"]
        thr = float(m3["threshold"])
        model_name = m3["model_name"]

        try:
            X_test, y_test = load_xy(
                f"{PROCESSED_DIR}/m3_X_test.parquet",
                f"{PROCESSED_DIR}/m3_y_test.npy"
            )
        except Exception as e:
            st.error("No se pudieron cargar los datos del módulo 3. Verifica data/processed/m3_X_test.parquet y m3_y_test.npy")
            st.exception(e)
            st.stop()

        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= thr).astype(int)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Modelo", model_name)
        c2.metric("Threshold", f"{thr:.3f}")
        c3.metric("Prob. alta conversión", f"{proba.mean():.3f}")
        c4.metric("% alta conv (pred)", f"{pred.mean():.3f}")

        st.divider()

        st.subheader("📈 Probabilidades de alta conversión (30 días)")
        st.plotly_chart(px.histogram(proba, nbins=40, title="High conversion prob (30 días)"),
                        use_container_width=True)

        st.subheader("🧩 Matriz de confusión")
        st.dataframe(confusion_df(y_test, pred), use_container_width=True)

        st.divider()

        st.subheader("🧠 Simular configuración de tienda")
        with st.form("m3_form"):
            a1, a2, a3, a4 = st.columns(4)
            payment_settings_enabled = a1.selectbox("payment_settings_enabled", [0, 1], index=1)
            hero_has_cta = a2.selectbox("hero_has_cta", [0, 1], index=1)
            hero_slides_count = a3.number_input("hero_slides_count", min_value=0, value=2)
            navigation_menu_items_count = a4.number_input("navigation_menu_items_count", min_value=0, value=5)

            b1, b2, b3, b4 = st.columns(4)
            has_custom_logo = b1.selectbox("has_custom_logo", [0, 1], index=1)
            has_cover_image = b2.selectbox("has_cover_image", [0, 1], index=1)
            products_with_image_pct = b3.number_input("products_with_image_pct", min_value=0.0, max_value=1.0, value=0.80)
            products_with_description_pct = b4.number_input("products_with_description_pct", min_value=0.0, max_value=1.0, value=0.60)

            c1_, c2_, c3_, c4_ = st.columns(4)
            total_products_active = c1_.number_input("total_products_active", min_value=1, value=60)
            products_with_discount_pct = c2_.number_input("products_with_discount_pct", min_value=0.0, max_value=1.0, value=0.25)
            avg_images_per_product = c3_.number_input("avg_images_per_product", min_value=0.0, value=2.5)
            plan_id = c4_.selectbox("plan_id", ["free", "basic", "pro", "enterprise"], index=1)

            d1, d2 = st.columns(2)
            template_slug = d1.selectbox("template_slug", ["Minimal", "Classic", "NikeStyle", "DarkLuxury", "Valentine", "Academy"], index=0)
            business_type_slug = d2.selectbox("business_type_slug", ["store", "appointments", "restaurant"], index=0)

            submit3 = st.form_submit_button("Evaluar diseño")

        if submit3:
            new = {c: None for c in X_test.columns}
            for k, v in {
                "payment_settings_enabled": payment_settings_enabled,
                "hero_has_cta": hero_has_cta,
                "hero_slides_count": hero_slides_count,
                "navigation_menu_items_count": navigation_menu_items_count,
                "has_custom_logo": has_custom_logo,
                "has_cover_image": has_cover_image,
                "products_with_image_pct": products_with_image_pct,
                "products_with_description_pct": products_with_description_pct,
                "total_products_active": total_products_active,
                "products_with_discount_pct": products_with_discount_pct,
                "avg_images_per_product": avg_images_per_product,
            }.items():
                if k in new:
                    new[k] = v

            if "plan_id" in new: new["plan_id"] = plan_id
            if "template_slug" in new: new["template_slug"] = template_slug
            if "business_type_slug" in new: new["business_type_slug"] = business_type_slug

            new_df = pd.DataFrame([new])
            p = float(pipe.predict_proba(new_df)[:, 1][0])
            yhat = int(p >= thr)

            st.success(f"Probabilidad de alta conversión (30 días): {p:.3f}")
            st.write("Predicción:", "ALTA ✅" if yhat else "MEDIA/BAJA ⚠️")

            rec = []
            if payment_settings_enabled == 0:
                rec.append("Activa pagos online: reduce fricción y sube conversión.")
            if hero_has_cta == 0:
                rec.append("Agrega CTA en el hero: 'Comprar ahora', 'Ver catálogo', 'Escríbenos'.")
            if navigation_menu_items_count > 9:
                rec.append("Tu menú está largo: reduce a 5–8 opciones para evitar confusión.")
            if products_with_image_pct < 0.75:
                rec.append("Sube fotos: apunta a 80%+ de productos con imagen.")
            if products_with_description_pct < 0.55:
                rec.append("Mejora descripciones: beneficios, medidas, garantía, compatibilidad.")
            if avg_images_per_product < 2.0:
                rec.append("Sube a 2–4 imágenes por producto (mínimo 2).")
            if not rec:
                rec.append("Config sólida: ahora prueba A/B con template, topbar y banners.")

            st.subheader("✅ Recomendaciones")
            for r in rec:
                st.write("•", r)

    tab_idx += 1


# =========================
# TAB 4 — Module 4 Content/Engagement
# =========================
if m4 is not None:
    with tabs[tab_idx]:
        st.header("🟧 Módulo 4 — Contenido/Engagement: Crecimiento + Recomendaciones")

        pipe = m4["pipeline"]
        thr = float(m4["threshold"])
        model_name = m4["model_name"]

        try:
            X_test, y_test = load_xy(
                f"{PROCESSED_DIR}/m4_X_test.parquet",
                f"{PROCESSED_DIR}/m4_y_test.npy"
            )
        except Exception as e:
            st.error("No se pudieron cargar los datos del módulo 4. Verifica data/processed/m4_X_test.parquet y m4_y_test.npy")
            st.exception(e)
            st.stop()

        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= thr).astype(int)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Modelo", model_name)
        c2.metric("Threshold", f"{thr:.3f}")
        c3.metric("Prob. crecimiento", f"{proba.mean():.3f}")
        c4.metric("% crecimiento (pred)", f"{pred.mean():.3f}")

        st.divider()

        st.subheader("📈 Probabilidad de crecimiento próxima semana")
        st.plotly_chart(px.histogram(proba, nbins=40, title="Growth probability (próxima semana)"),
                        use_container_width=True)

        st.subheader("🧩 Matriz de confusión")
        st.dataframe(confusion_df(y_test, pred), use_container_width=True)

        st.divider()

        st.subheader("🧠 Simular estrategia de contenido")
        with st.form("m4_form"):
            a1, a2, a3, a4 = st.columns(4)
            posts_count_30d = a1.number_input("posts_count_30d", min_value=0, value=10)
            stories_count_30d = a2.number_input("stories_count_30d", min_value=0, value=20)
            pct_posts_with_video = a3.number_input("pct_posts_with_video", min_value=0.0, max_value=1.0, value=0.50)
            days_between_posts = a4.number_input("days_between_posts", min_value=0.3, value=3.0)

            b1, b2, b3, b4 = st.columns(4)
            avg_views_per_post = b1.number_input("avg_views_per_post", min_value=0.0, value=1500.0)
            avg_likes_per_post = b2.number_input("avg_likes_per_post", min_value=0.0, value=60.0)
            avg_comments_per_post = b3.number_input("avg_comments_per_post", min_value=0.0, value=6.0)
            avg_rating = b4.number_input("avg_rating", min_value=1.0, max_value=5.0, value=4.5)

            c1_, c2_, c3_, c4_ = st.columns(4)
            pct_1_2_star = c1_.number_input("pct_1_2_star", min_value=0.0, max_value=1.0, value=0.05)
            reviews_with_photo_pct = c2_.number_input("reviews_with_photo_pct", min_value=0.0, max_value=1.0, value=0.20)
            post_engagement_rate = c3_.number_input("post_engagement_rate", min_value=0.0001, max_value=0.30, value=0.04)
            business_type_slug = c4_.selectbox("business_type_slug", ["store", "appointments", "restaurant"], index=0)

            d1, d2, d3, d4 = st.columns(4)
            has_instagram = d1.selectbox("has_instagram", [0, 1], index=1)
            has_tiktok = d2.selectbox("has_tiktok", [0, 1], index=1)
            has_facebook = d3.selectbox("has_facebook", [0, 1], index=1)
            has_whatsapp = d4.selectbox("has_whatsapp", [0, 1], index=1)

            submit4 = st.form_submit_button("Evaluar crecimiento")

        if submit4:
            new = {c: None for c in X_test.columns}

            for k, v in {
                "posts_count_30d": posts_count_30d,
                "stories_count_30d": stories_count_30d,
                "pct_posts_with_video": pct_posts_with_video,
                "days_between_posts": days_between_posts,
                "avg_views_per_post": avg_views_per_post,
                "avg_likes_per_post": avg_likes_per_post,
                "avg_comments_per_post": avg_comments_per_post,
                "post_engagement_rate": post_engagement_rate,
                "avg_rating": avg_rating,
                "pct_1_2_star": pct_1_2_star,
                "reviews_with_photo_pct": reviews_with_photo_pct,
                "has_instagram": has_instagram,
                "has_tiktok": has_tiktok,
                "has_facebook": has_facebook,
                "has_whatsapp": has_whatsapp,
            }.items():
                if k in new:
                    new[k] = v

            if "business_type_slug" in new:
                new["business_type_slug"] = business_type_slug

            new_df = pd.DataFrame([new])
            p = float(pipe.predict_proba(new_df)[:, 1][0])
            yhat = int(p >= thr)

            st.success(f"Probabilidad de crecimiento próxima semana: {p:.3f}")
            st.write("Predicción:", "CRECE ✅" if yhat else "ESTABLE/BAJA ⚠️")

            rec = []
            if pct_posts_with_video < 0.45:
                rec.append("Sube el % de video: apunta a 50–70% (mejor alcance orgánico).")
            if days_between_posts > 4:
                rec.append("Publicas muy espaciado: reduce a 2–3 días entre posts.")
            if stories_count_30d < 12:
                rec.append("Aumenta stories: 1 diaria mínimo (pruebas, ofertas, detrás de cámaras).")
            if post_engagement_rate < 0.03:
                rec.append("Mejora engagement: CTA en captions, preguntas, promos, carruseles educativos.")
            if pct_1_2_star > 0.10:
                rec.append("Hay muchas reseñas malas: responde rápido, mejora tiempos y claridad de entrega.")
            if avg_rating < 4.2:
                rec.append("Mejora reputación: solicita reseñas a clientes felices y muestra prueba social.")
            if has_whatsapp == 0:
                rec.append("Activa WhatsApp: reduce fricción y sube respuesta inmediata.")

            if not rec:
                rec.append("Estrategia sólida: haz A/B test de formato (reels vs carrusel) y horario de publicación.")

            st.subheader("✅ Recomendaciones")
            for r in rec:
                st.write("•", r)