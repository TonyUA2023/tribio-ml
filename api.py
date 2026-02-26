"""
TRIBIO IA — REST API
Motor de predicción ML expuesto como servicio HTTP (FastAPI).
Versión: 1.0.0

REST API (Hugging Face Spaces):
  https://tonyua-tribio.hf.space

Dashboard visual (Streamlit):
  https://tribio-ml-r5btzwfrf9z8eg3pqnkw2h.streamlit.app/

Desplegado con Docker en HF Spaces (puerto interno 7860).
"""

import os
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ART_DIR = os.getenv("TRIBIO_ARTIFACTS_DIR", "artifacts")

MODELS: dict = {}


def _load_bundle(name: str):
    path = os.path.join(ART_DIR, f"model_{name}.joblib")
    if not os.path.exists(path):
        return None
    return load(path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load models into memory
    MODELS["m1"] = _load_bundle("module1")
    MODELS["m2"] = _load_bundle("module2")
    MODELS["m3"] = _load_bundle("module3")
    MODELS["m4"] = _load_bundle("module4")
    loaded = [k for k, v in MODELS.items() if v is not None]
    print(f"[TRIBIO API] Modelos cargados: {loaded}")
    yield
    MODELS.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="TRIBIO IA — Prediction API",
    description=(
        "Motor de predicción ML de Tribio. Expone 4 módulos de IA:\n\n"
        "- **Módulo 1**: Predicción de ventas del producto (próximos 7 días)\n"
        "- **Módulo 2**: Riesgo de churn de clientes (próximos 30 días)\n"
        "- **Módulo 3**: Probabilidad de alta conversión (diseño/config)\n"
        "- **Módulo 4**: Predicción de crecimiento por contenido/engagement\n\n"
        "Cada endpoint devuelve `probability` (0-1), `prediction` (0/1) y `recommendations` accionables.\n\n"
        "**REST API**: https://tonyua-tribio.hf.space\n\n"
        "**Dashboard visual**: https://tribio-ml-r5btzwfrf9z8eg3pqnkw2h.streamlit.app/"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Ajusta en producción al dominio de Laravel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_model_cols(bundle: dict) -> list[str]:
    """Extrae las columnas exactas que espera el pipeline del modelo."""
    # Primero intenta feature_columns guardado en el bundle
    if "feature_columns" in bundle:
        return list(bundle["feature_columns"])
    # Si no, extrae del ColumnTransformer (M3, M4)
    pipe = bundle["pipeline"]
    pre = pipe.steps[0][1]
    if hasattr(pre, "transformers"):
        cols = []
        for _, _, c in pre.transformers:
            cols.extend(c)
        return cols
    return []


def _predict(bundle_key: str, row: dict) -> tuple[float, int]:
    """
    Devuelve (probability, prediction) usando el pipeline y threshold del bundle.
    Construye el DataFrame con las columnas exactas del modelo, llenando con None
    las columnas que no vienen en el request (el SimpleImputer las maneja).
    """
    bundle = MODELS.get(bundle_key)
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo {bundle_key} no disponible. Verifica artifacts/.",
        )
    pipe = bundle["pipeline"]
    thr = float(bundle["threshold"])

    # Construir fila con exactamente las columnas del modelo
    model_cols = _get_model_cols(bundle)
    if model_cols:
        aligned = {col: row.get(col, None) for col in model_cols}
    else:
        aligned = row

    df = pd.DataFrame([aligned])
    prob = float(pipe.predict_proba(df)[:, 1][0])
    pred = int(prob >= thr)
    return prob, pred


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

# --- Module 1 ---
class SalesInput(BaseModel):
    price: float = Field(..., example=55.0, description="Precio del producto")
    discount_pct: float = Field(0.0, example=10.0, description="Porcentaje de descuento (0-90)")
    stock: int = Field(30, example=30, description="Stock disponible (-1 = ilimitado)")
    images_count: int = Field(3, example=3, description="Cantidad de imágenes del producto")
    description_length: int = Field(400, example=400, description="Longitud de la descripción en caracteres")
    featured: int = Field(0, example=0, description="Producto destacado (0/1)")
    payment_settings_enabled: int = Field(1, example=1, description="Pagos online habilitados (0/1)")
    has_whatsapp: int = Field(1, example=1, description="Tiene WhatsApp de contacto (0/1)")
    plan_id: str = Field("basic", example="basic", description="Plan del negocio: free|basic|pro|enterprise")
    business_type_slug: str = Field("store", example="store", description="Tipo de negocio: store|appointments|restaurant")
    business_category_slug: str = Field("retail", example="retail", description="Categoría del negocio")


class SalesOutput(BaseModel):
    probability: float = Field(..., description="Probabilidad de venta en los próximos 7 días (0-1)")
    prediction: int = Field(..., description="1 = venderá, 0 = no venderá")
    label: str
    threshold: float
    recommendations: list[str]


# --- Module 2 ---
class ChurnInput(BaseModel):
    days_since_last_order: int = Field(..., example=25, description="Días desde el último pedido")
    total_orders_paid: int = Field(..., example=2, description="Total de órdenes pagadas")
    avg_order_value: float = Field(..., example=70.0, description="Valor promedio de pedidos")
    cancellation_rate: float = Field(0.05, example=0.05, description="Tasa de cancelación (0-1)")
    profile_visits_count: int = Field(10, example=10, description="Visitas al perfil del negocio")
    link_click_to_order_ratio: float = Field(1.2, example=1.2, description="Ratio clics a pedidos")
    preferred_payment_method: str = Field("cash", example="cash", description="Método de pago preferido: cash|card|transfer")
    preferred_notification: str = Field("whatsapp", example="whatsapp", description="Canal de notificación: email|whatsapp|sms")


class ChurnOutput(BaseModel):
    probability: float = Field(..., description="Probabilidad de churn en los próximos 30 días (0-1)")
    prediction: int = Field(..., description="1 = en riesgo, 0 = estable")
    label: str
    threshold: float
    recommendations: list[str]


# --- Module 3 ---
class DesignInput(BaseModel):
    payment_settings_enabled: int = Field(1, example=1, description="Pagos online habilitados (0/1)")
    hero_has_cta: int = Field(1, example=1, description="El hero tiene CTA (0/1)")
    hero_slides_count: int = Field(2, example=2, description="Cantidad de slides en el hero")
    navigation_menu_items_count: int = Field(5, example=5, description="Items en el menú de navegación")
    has_custom_logo: int = Field(1, example=1, description="Tiene logo personalizado (0/1)")
    has_cover_image: int = Field(1, example=1, description="Tiene imagen de portada (0/1)")
    products_with_image_pct: float = Field(0.80, example=0.80, description="% de productos con imagen (0-1)")
    products_with_description_pct: float = Field(0.60, example=0.60, description="% de productos con descripción (0-1)")
    total_products_active: int = Field(60, example=60, description="Total de productos activos")
    products_with_discount_pct: float = Field(0.25, example=0.25, description="% de productos con descuento (0-1)")
    avg_images_per_product: float = Field(2.5, example=2.5, description="Promedio de imágenes por producto")
    plan_id: str = Field("basic", example="basic", description="Plan del negocio: free|basic|pro|enterprise")
    template_slug: str = Field("Minimal", example="Minimal", description="Plantilla: Minimal|Classic|NikeStyle|DarkLuxury|Valentine|Academy")
    business_type_slug: str = Field("store", example="store", description="Tipo de negocio: store|appointments|restaurant")


class DesignOutput(BaseModel):
    probability: float = Field(..., description="Probabilidad de alta conversión en los próximos 30 días (0-1)")
    prediction: int = Field(..., description="1 = alta conversión, 0 = media/baja")
    label: str
    threshold: float
    recommendations: list[str]


# --- Module 4 ---
class GrowthInput(BaseModel):
    posts_count_30d: int = Field(10, example=10, description="Posts publicados en últimos 30 días")
    stories_count_30d: int = Field(20, example=20, description="Stories publicadas en últimos 30 días")
    pct_posts_with_video: float = Field(0.50, example=0.50, description="% de posts con video (0-1)")
    days_between_posts: float = Field(3.0, example=3.0, description="Días promedio entre publicaciones")
    avg_views_per_post: float = Field(1500.0, example=1500.0, description="Promedio de vistas por post")
    avg_likes_per_post: float = Field(60.0, example=60.0, description="Promedio de likes por post")
    avg_comments_per_post: float = Field(6.0, example=6.0, description="Promedio de comentarios por post")
    avg_rating: float = Field(4.5, example=4.5, description="Calificación promedio del negocio (1-5)")
    pct_1_2_star: float = Field(0.05, example=0.05, description="% de reseñas de 1-2 estrellas (0-1)")
    reviews_with_photo_pct: float = Field(0.20, example=0.20, description="% de reseñas con foto (0-1)")
    post_engagement_rate: float = Field(0.04, example=0.04, description="Tasa de engagement por post (0-0.30)")
    business_type_slug: str = Field("store", example="store", description="Tipo de negocio: store|appointments|restaurant")
    has_instagram: int = Field(1, example=1, description="Tiene Instagram activo (0/1)")
    has_tiktok: int = Field(1, example=1, description="Tiene TikTok activo (0/1)")
    has_facebook: int = Field(1, example=1, description="Tiene Facebook activo (0/1)")
    has_whatsapp: int = Field(1, example=1, description="Tiene WhatsApp activo (0/1)")


class GrowthOutput(BaseModel):
    probability: float = Field(..., description="Probabilidad de crecimiento la próxima semana (0-1)")
    prediction: int = Field(..., description="1 = crecerá, 0 = estable/baja")
    label: str
    threshold: float
    recommendations: list[str]


# --- Health / Status ---
class HealthOutput(BaseModel):
    status: str
    models_loaded: dict[str, bool]
    version: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    """Bienvenida rápida."""
    return {
        "message": "TRIBIO IA Prediction API",
        "docs": "https://tonyua-tribio.hf.space/docs",
        "api_base": "https://tonyua-tribio.hf.space",
        "dashboard": "https://tribio-ml-r5btzwfrf9z8eg3pqnkw2h.streamlit.app/",
        "version": "1.0.0",
    }


@app.get("/health", response_model=HealthOutput, tags=["Health"])
def health():
    """Verifica el estado del servicio y qué modelos están cargados."""
    return HealthOutput(
        status="ok",
        models_loaded={k: v is not None for k, v in MODELS.items()},
        version="1.0.0",
    )


# ---------------------------------------------------------------------------
# Scoring propio para M1 (el modelo ML fue entrenado con datos históricos
# de ventas que no están disponibles en tiempo real; usamos un scoring
# basado en los campos de configuración del producto que sí tenemos)
# ---------------------------------------------------------------------------
def _score_sales(data: "SalesInput") -> tuple[float, float]:
    """
    Calcula un score de vendibilidad (0-1) basado en la configuración del producto.
    Devuelve (score, threshold).
    """
    score = 0.0

    # Precio competitivo (penaliza extremos)
    if 5 <= data.price <= 200:
        score += 0.20
    elif data.price < 5 or data.price > 500:
        score += 0.05
    else:
        score += 0.12

    # Descuento activo
    if data.discount_pct >= 10:
        score += 0.15
    elif data.discount_pct > 0:
        score += 0.08

    # Stock disponible
    if data.stock == 0:
        score += 0.00   # sin stock = no vende
    elif data.stock == -1 or data.stock >= 10:
        score += 0.15
    else:
        score += 0.08

    # Imágenes
    if data.images_count >= 4:
        score += 0.15
    elif data.images_count >= 2:
        score += 0.10
    elif data.images_count == 1:
        score += 0.04
    # 0 imágenes = 0 puntos

    # Descripción
    if data.description_length >= 400:
        score += 0.10
    elif data.description_length >= 150:
        score += 0.06
    elif data.description_length >= 50:
        score += 0.02

    # Pagos online habilitados
    if data.payment_settings_enabled == 1:
        score += 0.10

    # WhatsApp de contacto
    if data.has_whatsapp == 1:
        score += 0.05

    # Producto destacado
    if data.featured == 1:
        score += 0.05

    # Plan del negocio
    plan_bonus = {"free": 0.0, "basic": 0.03, "pro": 0.05, "enterprise": 0.05}
    score += plan_bonus.get(data.plan_id, 0.02)

    return round(min(score, 1.0), 4), 0.55


# ── Módulo 1: Ventas ────────────────────────────────────────────────────────
@app.post(
    "/predict/sales",
    response_model=SalesOutput,
    tags=["Módulo 1 — Ventas"],
    summary="Predicción de ventas (próximos 7 días)",
)
def predict_sales(data: SalesInput):
    """
    Predice si un **producto venderá en los próximos 7 días** dada su configuración actual.

    - `probability`: score de 0 a 1
    - `prediction`: 1 = vende, 0 = no vende
    - `recommendations`: acciones concretas para mejorar el producto
    """
    prob, thr = _score_sales(data)
    pred = int(prob >= thr)

    rec = []
    if data.discount_pct == 0:
        rec.append("Prueba un descuento suave (5-10%) o envío gratis para empujar la primera compra.")
    if data.images_count < 3:
        rec.append("Mejora la ficha: agrega 3-5 fotos reales (frontal, detalle, uso).")
    if data.description_length < 150:
        rec.append("Mejora la descripción: beneficios, compatibilidad, garantía, FAQs (mínimo 150 chars).")
    if data.stock == 0:
        rec.append("Stock en 0: reabastece o muestra alternativas para no cortar el embudo.")
    if data.payment_settings_enabled == 0:
        rec.append("Activa pagos online: reduce fricción y mejora conversión.")
    if not rec:
        rec.append("Ficha sólida: prueba A/B de precio y CTA para subir ticket y conversión.")

    return SalesOutput(
        probability=round(prob, 4),
        prediction=pred,
        label="venderá ✅" if pred == 1 else "no venderá ❌",
        threshold=round(thr, 4),
        recommendations=rec,
    )


# ── Módulo 2: Churn ─────────────────────────────────────────────────────────
@app.post(
    "/predict/churn",
    response_model=ChurnOutput,
    tags=["Módulo 2 — Clientes (Churn)"],
    summary="Predicción de churn de clientes (próximos 30 días)",
)
def predict_churn(data: ChurnInput):
    """
    Predice si un **cliente está en riesgo de churn en los próximos 30 días**.

    - `probability`: score de riesgo de 0 a 1
    - `prediction`: 1 = en riesgo, 0 = estable
    - `recommendations`: estrategias de retención personalizadas
    """
    # M2 fue entrenado con dataset Telco — mapeamos los campos disponibles
    # El resto de columnas se completan con None y el SimpleImputer las imputa
    row = {
        "MonthlyCharges": data.avg_order_value,
        "TotalCharges": float(data.total_orders_paid * data.avg_order_value),
        "tenure": max(0, 90 - data.days_since_last_order),  # proxy de antigüedad
        "PaymentMethod": "Electronic check" if data.preferred_payment_method == "card"
                         else "Mailed check",
        "Contract": "Month-to-month" if data.total_orders_paid <= 3 else "One year",
        "PaperlessBilling": 1 if data.preferred_notification in ("email", "whatsapp") else 0,
    }

    prob, pred = _predict("m2", row)

    bundle = MODELS["m2"]
    thr = float(bundle["threshold"])

    rec = []
    if data.days_since_last_order >= 30:
        rec.append("Enviar WhatsApp personalizado con oferta limitada (alta recencia).")
    if data.total_orders_paid <= 1:
        rec.append("Activar campaña de bienvenida / primera recompra (cliente nuevo).")
    if data.cancellation_rate >= 0.12:
        rec.append("Reducir fricciones: claridad de precios, tiempos de entrega, métodos de pago.")
    if data.profile_visits_count >= 15 and data.total_orders_paid == 0:
        rec.append("Alta intención sin compra: cupón 5-10% + asistencia rápida por WhatsApp.")
    if data.preferred_payment_method == "cash":
        rec.append("Ofrecer transferencia/tarjeta con incentivo (reduce abandono).")
    if not rec:
        rec.append("Cliente saludable: upsell suave o programa de fidelidad para aumentar LTV.")

    return ChurnOutput(
        probability=round(prob, 4),
        prediction=pred,
        label="en riesgo ⚠️" if pred == 1 else "estable ✅",
        threshold=round(thr, 4),
        recommendations=rec,
    )


# ── Módulo 3: Diseño ─────────────────────────────────────────────────────────
@app.post(
    "/predict/design",
    response_model=DesignOutput,
    tags=["Módulo 3 — Diseño/Config"],
    summary="Predicción de alta conversión según configuración de la tienda",
)
def predict_design(data: DesignInput):
    """
    Predice si la **configuración actual de la tienda generará alta conversión en los próximos 30 días**.

    - `probability`: score de 0 a 1
    - `prediction`: 1 = alta conversión, 0 = media/baja
    - `recommendations`: mejoras concretas al diseño y configuración
    """
    row = data.model_dump()
    prob, pred = _predict("m3", row)

    bundle = MODELS["m3"]
    thr = float(bundle["threshold"])

    rec = []
    if data.payment_settings_enabled == 0:
        rec.append("Activa pagos online: reduce fricción y sube conversión.")
    if data.hero_has_cta == 0:
        rec.append("Agrega CTA en el hero: 'Comprar ahora', 'Ver catálogo', 'Escríbenos'.")
    if data.navigation_menu_items_count > 9:
        rec.append("Tu menú está largo: reduce a 5-8 opciones para evitar confusión.")
    if data.products_with_image_pct < 0.75:
        rec.append("Sube fotos: apunta a 80%+ de productos con imagen.")
    if data.products_with_description_pct < 0.55:
        rec.append("Mejora descripciones: beneficios, medidas, garantía, compatibilidad.")
    if data.avg_images_per_product < 2.0:
        rec.append("Sube a 2-4 imágenes por producto (mínimo 2).")
    if not rec:
        rec.append("Config sólida: ahora prueba A/B con template, topbar y banners.")

    return DesignOutput(
        probability=round(prob, 4),
        prediction=pred,
        label="alta conversión ✅" if pred == 1 else "media/baja ⚠️",
        threshold=round(thr, 4),
        recommendations=rec,
    )


# ── Módulo 4: Crecimiento ────────────────────────────────────────────────────
@app.post(
    "/predict/growth",
    response_model=GrowthOutput,
    tags=["Módulo 4 — Contenido/Engagement"],
    summary="Predicción de crecimiento según estrategia de contenido",
)
def predict_growth(data: GrowthInput):
    """
    Predice si el **negocio crecerá la próxima semana** basado en su estrategia de contenido y engagement.

    - `probability`: score de 0 a 1
    - `prediction`: 1 = crecerá, 0 = estable/baja
    - `recommendations`: estrategia de contenido accionable
    """
    row = data.model_dump()
    prob, pred = _predict("m4", row)

    bundle = MODELS["m4"]
    thr = float(bundle["threshold"])

    rec = []
    if data.pct_posts_with_video < 0.45:
        rec.append("Sube el % de video: apunta a 50-70% (mejor alcance orgánico).")
    if data.days_between_posts > 4:
        rec.append("Publicas muy espaciado: reduce a 2-3 días entre posts.")
    if data.stories_count_30d < 12:
        rec.append("Aumenta stories: 1 diaria mínimo (pruebas, ofertas, detrás de cámaras).")
    if data.post_engagement_rate < 0.03:
        rec.append("Mejora engagement: CTA en captions, preguntas, promos, carruseles educativos.")
    if data.pct_1_2_star > 0.10:
        rec.append("Hay muchas reseñas malas: responde rápido, mejora tiempos y claridad de entrega.")
    if data.avg_rating < 4.2:
        rec.append("Mejora reputación: solicita reseñas a clientes felices y muestra prueba social.")
    if data.has_whatsapp == 0:
        rec.append("Activa WhatsApp: reduce fricción y sube respuesta inmediata.")
    if not rec:
        rec.append("Estrategia sólida: haz A/B test de formato (reels vs carrusel) y horario de publicación.")

    return GrowthOutput(
        probability=round(prob, 4),
        prediction=pred,
        label="crecerá ✅" if pred == 1 else "estable/baja ⚠️",
        threshold=round(thr, 4),
        recommendations=rec,
    )
