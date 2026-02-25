# TRIBIO IA — Documentación de la API REST v1.0.0

> Motor de predicción ML para integración con sistemas externos (Laravel u otros).

---

## Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────────────┐
│  TRIBIO IA — Dos servicios independientes                       │
│                                                                 │
│  1. Dashboard visual (Streamlit Cloud — solo lectura/demo)      │
│     https://tribio-ml-r5btzwfrf9z8eg3pqnkw2h.streamlit.app/   │
│                                                                 │
│  2. REST API (FastAPI — para integración programática)          │
│     https://tonyua-tribio.hf.space   ← debes desplegar este servicio  │
└─────────────────────────────────────────────────────────────────┘
```

> **Importante**: El dashboard de Streamlit es una interfaz visual de demostración.
> Para consumir las predicciones desde Laravel u otro sistema, debes desplegar
> la REST API (`api.py`) en un servidor con Python. Son dos servicios distintos.

---

## Índice

1. [Descripción general](#1-descripción-general)
2. [Instalación y arranque de la REST API](#2-instalación-y-arranque-de-la-rest-api)
3. [Autenticación](#3-autenticación)
4. [Endpoints](#4-endpoints)
   - [GET /health](#get-health)
   - [POST /predict/sales](#post-predictsales)
   - [POST /predict/churn](#post-predictchurn)
   - [POST /predict/design](#post-predictdesign)
   - [POST /predict/growth](#post-predictgrowth)
5. [Respuesta estándar](#5-respuesta-estándar)
6. [Códigos de error](#6-códigos-de-error)
7. [Integración con Laravel](#7-integración-con-laravel)
8. [Despliegue en producción](#8-despliegue-en-producción)

---

## 1. Descripción general

La API REST expone **4 módulos de predicción ML** entrenados sobre datos reales de e-commerce:

| Módulo | Endpoint | Pregunta que responde | Horizonte |
|--------|----------|-----------------------|-----------|
| M1 — Ventas | `POST /predict/sales` | ¿Este producto venderá? | 7 días |
| M2 — Clientes | `POST /predict/churn` | ¿Este cliente hará churn? | 30 días |
| M3 — Diseño | `POST /predict/design` | ¿Esta tienda tendrá alta conversión? | 30 días |
| M4 — Contenido | `POST /predict/growth` | ¿El negocio crecerá? | 7 días |

### Performance de los modelos

| Módulo | Algoritmo | F1 | ROC-AUC | Muestras entrenamiento |
|--------|-----------|----|---------|------------------------|
| M1 Ventas | LogisticRegression | 0.99 | 0.86 | 218,698 |
| M2 Churn | LogisticRegression | 0.62 | 0.84 | 5,634 |
| M3 Diseño | LogisticRegression | 0.75 | 0.91 | — |
| M4 Contenido | LogisticRegression | — | 0.59 | — |

---

## 2. Instalación y arranque de la REST API

### Prerrequisitos
- Python 3.10+
- Modelos entrenados en `artifacts/` (`model_module1.joblib`, `model_module2.joblib`, etc.)
- Repositorio del proyecto: `tribio-ml/`

### Instalar dependencias

```bash
pip install -r requirements.txt
```

### Arrancar el servidor

**Desarrollo:**
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Producción:**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 2
```

### Verificar que funciona

```
GET https://tonyua-tribio.hf.space/health
GET https://tonyua-tribio.hf.space/docs        ← Swagger UI interactivo
GET https://tonyua-tribio.hf.space/redoc       ← ReDoc
```

### Variable de entorno (modelos en otra ruta)

```bash
export TRIBIO_ARTIFACTS_DIR=/ruta/absoluta/a/artifacts
uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## 3. Autenticación

La v1.0 **no requiere autenticación** (pensada para red interna entre servidores).

Para producción, agrega un API Key en el header de cada request:

```
X-API-Key: tu_clave_secreta
```

---

## 4. Endpoints

---

### GET /health

Verifica que el servicio está activo y qué modelos están cargados.

```http
GET https://tonyua-tribio.hf.space/health
```

**Response 200:**
```json
{
  "status": "ok",
  "models_loaded": {
    "m1": true,
    "m2": true,
    "m3": true,
    "m4": true
  },
  "version": "1.0.0"
}
```

---

### POST /predict/sales

**¿Este producto venderá en los próximos 7 días?**

```http
POST https://tonyua-tribio.hf.space/predict/sales
Content-Type: application/json
```

**Request Body:**

| Campo | Tipo | Requerido | Default | Descripción |
|-------|------|:---------:|---------|-------------|
| `price` | float | ✅ | — | Precio del producto |
| `discount_pct` | float | ❌ | 0.0 | Descuento aplicado (0–90) |
| `stock` | int | ❌ | 30 | Stock disponible (-1 = ilimitado) |
| `images_count` | int | ❌ | 3 | Cantidad de imágenes del producto |
| `description_length` | int | ❌ | 400 | Longitud de la descripción (caracteres) |
| `featured` | int | ❌ | 0 | Producto destacado: `0` o `1` |
| `payment_settings_enabled` | int | ❌ | 1 | Pagos online activos: `0` o `1` |
| `has_whatsapp` | int | ❌ | 1 | Tiene WhatsApp de contacto: `0` o `1` |
| `plan_id` | string | ❌ | `"basic"` | `free` · `basic` · `pro` · `enterprise` |
| `business_type_slug` | string | ❌ | `"store"` | `store` · `appointments` · `restaurant` |
| `business_category_slug` | string | ❌ | `"retail"` | Categoría libre (retail, food, health…) |

**Ejemplo:**
```json
{
  "price": 55.0,
  "discount_pct": 10.0,
  "stock": 30,
  "images_count": 4,
  "description_length": 450,
  "featured": 0,
  "payment_settings_enabled": 1,
  "has_whatsapp": 1,
  "plan_id": "basic",
  "business_type_slug": "store",
  "business_category_slug": "retail"
}
```

**Response 200:**
```json
{
  "probability": 0.8214,
  "prediction": 1,
  "label": "venderá ✅",
  "threshold": 0.42,
  "recommendations": [
    "Ficha sólida: prueba A/B de precio y CTA para subir ticket y conversión."
  ]
}
```

---

### POST /predict/churn

**¿Este cliente está en riesgo de churn en los próximos 30 días?**

```http
POST https://tonyua-tribio.hf.space/predict/churn
Content-Type: application/json
```

**Request Body:**

| Campo | Tipo | Requerido | Default | Descripción |
|-------|------|:---------:|---------|-------------|
| `days_since_last_order` | int | ✅ | — | Días desde el último pedido |
| `total_orders_paid` | int | ✅ | — | Total de órdenes pagadas |
| `avg_order_value` | float | ✅ | — | Valor promedio de pedidos |
| `cancellation_rate` | float | ❌ | 0.05 | Tasa de cancelación (0–1) |
| `profile_visits_count` | int | ❌ | 10 | Visitas al perfil del negocio |
| `link_click_to_order_ratio` | float | ❌ | 1.2 | Ratio clics / órdenes |
| `preferred_payment_method` | string | ❌ | `"cash"` | `cash` · `card` · `transfer` |
| `preferred_notification` | string | ❌ | `"whatsapp"` | `email` · `whatsapp` · `sms` |

**Ejemplo:**
```json
{
  "days_since_last_order": 25,
  "total_orders_paid": 2,
  "avg_order_value": 70.0,
  "cancellation_rate": 0.05,
  "profile_visits_count": 10,
  "preferred_payment_method": "cash",
  "preferred_notification": "whatsapp"
}
```

**Response 200:**
```json
{
  "probability": 0.6302,
  "prediction": 1,
  "label": "en riesgo ⚠️",
  "threshold": 0.51,
  "recommendations": [
    "Enviar WhatsApp personalizado con oferta limitada (alta recencia).",
    "Ofrecer transferencia/tarjeta con incentivo (reduce abandono)."
  ]
}
```

---

### POST /predict/design

**¿La configuración de esta tienda generará alta conversión en los próximos 30 días?**

```http
POST https://tonyua-tribio.hf.space/predict/design
Content-Type: application/json
```

**Request Body:**

| Campo | Tipo | Requerido | Default | Descripción |
|-------|------|:---------:|---------|-------------|
| `payment_settings_enabled` | int | ❌ | 1 | Pagos online activos: `0` o `1` |
| `hero_has_cta` | int | ❌ | 1 | Hero con botón CTA: `0` o `1` |
| `hero_slides_count` | int | ❌ | 2 | Número de slides en el hero |
| `navigation_menu_items_count` | int | ❌ | 5 | Items en el menú de navegación |
| `has_custom_logo` | int | ❌ | 1 | Logo personalizado: `0` o `1` |
| `has_cover_image` | int | ❌ | 1 | Imagen de portada: `0` o `1` |
| `products_with_image_pct` | float | ❌ | 0.80 | % productos con imagen (0–1) |
| `products_with_description_pct` | float | ❌ | 0.60 | % productos con descripción (0–1) |
| `total_products_active` | int | ❌ | 60 | Total productos activos |
| `products_with_discount_pct` | float | ❌ | 0.25 | % productos con descuento (0–1) |
| `avg_images_per_product` | float | ❌ | 2.5 | Promedio imágenes por producto |
| `plan_id` | string | ❌ | `"basic"` | `free` · `basic` · `pro` · `enterprise` |
| `template_slug` | string | ❌ | `"Minimal"` | `Minimal` · `Classic` · `NikeStyle` · `DarkLuxury` · `Valentine` · `Academy` |
| `business_type_slug` | string | ❌ | `"store"` | `store` · `appointments` · `restaurant` |

**Ejemplo:**
```json
{
  "payment_settings_enabled": 1,
  "hero_has_cta": 1,
  "hero_slides_count": 2,
  "navigation_menu_items_count": 5,
  "has_custom_logo": 1,
  "has_cover_image": 1,
  "products_with_image_pct": 0.85,
  "products_with_description_pct": 0.70,
  "total_products_active": 60,
  "products_with_discount_pct": 0.25,
  "avg_images_per_product": 3.0,
  "plan_id": "pro",
  "template_slug": "Classic",
  "business_type_slug": "store"
}
```

**Response 200:**
```json
{
  "probability": 0.7541,
  "prediction": 1,
  "label": "alta conversión ✅",
  "threshold": 0.45,
  "recommendations": [
    "Config sólida: ahora prueba A/B con template, topbar y banners."
  ]
}
```

---

### POST /predict/growth

**¿El negocio crecerá la próxima semana según su estrategia de contenido?**

```http
POST https://tonyua-tribio.hf.space/predict/growth
Content-Type: application/json
```

**Request Body:**

| Campo | Tipo | Requerido | Default | Descripción |
|-------|------|:---------:|---------|-------------|
| `posts_count_30d` | int | ❌ | 10 | Posts publicados (últimos 30 días) |
| `stories_count_30d` | int | ❌ | 20 | Stories publicadas (últimos 30 días) |
| `pct_posts_with_video` | float | ❌ | 0.50 | % posts con video (0–1) |
| `days_between_posts` | float | ❌ | 3.0 | Días promedio entre publicaciones |
| `avg_views_per_post` | float | ❌ | 1500.0 | Promedio vistas por post |
| `avg_likes_per_post` | float | ❌ | 60.0 | Promedio likes por post |
| `avg_comments_per_post` | float | ❌ | 6.0 | Promedio comentarios por post |
| `avg_rating` | float | ❌ | 4.5 | Calificación promedio (1–5) |
| `pct_1_2_star` | float | ❌ | 0.05 | % reseñas de 1–2 estrellas (0–1) |
| `reviews_with_photo_pct` | float | ❌ | 0.20 | % reseñas con foto (0–1) |
| `post_engagement_rate` | float | ❌ | 0.04 | Tasa de engagement (0–0.30) |
| `business_type_slug` | string | ❌ | `"store"` | `store` · `appointments` · `restaurant` |
| `has_instagram` | int | ❌ | 1 | Instagram activo: `0` o `1` |
| `has_tiktok` | int | ❌ | 1 | TikTok activo: `0` o `1` |
| `has_facebook` | int | ❌ | 1 | Facebook activo: `0` o `1` |
| `has_whatsapp` | int | ❌ | 1 | WhatsApp activo: `0` o `1` |

**Ejemplo:**
```json
{
  "posts_count_30d": 14,
  "stories_count_30d": 28,
  "pct_posts_with_video": 0.60,
  "days_between_posts": 2.1,
  "avg_views_per_post": 2000.0,
  "avg_likes_per_post": 90.0,
  "avg_comments_per_post": 10.0,
  "avg_rating": 4.7,
  "pct_1_2_star": 0.03,
  "reviews_with_photo_pct": 0.25,
  "post_engagement_rate": 0.05,
  "business_type_slug": "store",
  "has_instagram": 1,
  "has_tiktok": 1,
  "has_facebook": 0,
  "has_whatsapp": 1
}
```

**Response 200:**
```json
{
  "probability": 0.7102,
  "prediction": 1,
  "label": "crecerá ✅",
  "threshold": 0.27,
  "recommendations": [
    "Estrategia sólida: haz A/B test de formato (reels vs carrusel) y horario de publicación."
  ]
}
```

---

## 5. Respuesta estándar

Todos los endpoints de predicción devuelven la misma estructura:

```json
{
  "probability": 0.0,
  "prediction": 0,
  "label": "string",
  "threshold": 0.0,
  "recommendations": ["string"]
}
```

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `probability` | float (0–1) | Score de probabilidad del modelo |
| `prediction` | int (0 o 1) | Resultado binario según threshold optimizado |
| `label` | string | Resultado en texto legible |
| `threshold` | float | Umbral de decisión del modelo |
| `recommendations` | string[] | Lista de acciones accionables |

**Labels por módulo:**

| Módulo | prediction=1 | prediction=0 |
|--------|-------------|-------------|
| Sales | `"venderá ✅"` | `"no venderá ❌"` |
| Churn | `"en riesgo ⚠️"` | `"estable ✅"` |
| Design | `"alta conversión ✅"` | `"media/baja ⚠️"` |
| Growth | `"crecerá ✅"` | `"estable/baja ⚠️"` |

---

## 6. Códigos de error

| Código | Significado | Causa típica |
|--------|-------------|--------------|
| `200` | OK | Predicción exitosa |
| `422` | Unprocessable Entity | Campo requerido faltante o tipo incorrecto |
| `503` | Service Unavailable | Modelo `.joblib` no encontrado en `artifacts/` |
| `500` | Internal Server Error | Error inesperado del servidor |

**Ejemplo de error 422:**
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "price"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```

---

## 7. Integración con Laravel

### Configuración en `.env`

```env
TRIBIO_API_URL=https://tonyua-tribio.hf.space
```

### `config/services.php`

```php
'tribio_ml' => [
    'url' => env('TRIBIO_API_URL', 'https://tonyua-tribio.hf.space'),
],
```

### Service class — `app/Services/TribioMLService.php`

```php
<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;
use Illuminate\Http\Client\RequestException;

class TribioMLService
{
    protected string $baseUrl;

    public function __construct()
    {
        $this->baseUrl = rtrim(config('services.tribio_ml.url'), '/');
    }

    /** Verifica el estado del servicio ML */
    public function health(): array
    {
        return Http::get("{$this->baseUrl}/health")->json();
    }

    /**
     * M1 — Predicción de ventas de un producto (próximos 7 días)
     *
     * Campos requeridos: price (float)
     * Campos opcionales: discount_pct, stock, images_count, description_length,
     *   featured (0|1), payment_settings_enabled (0|1), has_whatsapp (0|1),
     *   plan_id (free|basic|pro|enterprise),
     *   business_type_slug (store|appointments|restaurant),
     *   business_category_slug (string libre)
     */
    public function predictSales(array $data): array
    {
        return $this->post('/predict/sales', $data);
    }

    /**
     * M2 — Riesgo de churn de un cliente (próximos 30 días)
     *
     * Campos requeridos: days_since_last_order (int), total_orders_paid (int),
     *   avg_order_value (float)
     * Campos opcionales: cancellation_rate, profile_visits_count,
     *   link_click_to_order_ratio,
     *   preferred_payment_method (cash|card|transfer),
     *   preferred_notification (email|whatsapp|sms)
     */
    public function predictChurn(array $data): array
    {
        return $this->post('/predict/churn', $data);
    }

    /**
     * M3 — Alta conversión según diseño/config de la tienda (próximos 30 días)
     *
     * Todos los campos son opcionales. Envía los que tengas disponibles.
     * Campos: payment_settings_enabled, hero_has_cta, hero_slides_count,
     *   navigation_menu_items_count, has_custom_logo, has_cover_image,
     *   products_with_image_pct, products_with_description_pct,
     *   total_products_active, products_with_discount_pct, avg_images_per_product,
     *   plan_id, template_slug, business_type_slug
     */
    public function predictDesign(array $data): array
    {
        return $this->post('/predict/design', $data);
    }

    /**
     * M4 — Predicción de crecimiento por contenido/engagement (próxima semana)
     *
     * Todos los campos son opcionales. Envía los que tengas disponibles.
     * Campos: posts_count_30d, stories_count_30d, pct_posts_with_video,
     *   days_between_posts, avg_views_per_post, avg_likes_per_post,
     *   avg_comments_per_post, avg_rating, pct_1_2_star, reviews_with_photo_pct,
     *   post_engagement_rate, business_type_slug,
     *   has_instagram, has_tiktok, has_facebook, has_whatsapp (0|1)
     */
    public function predictGrowth(array $data): array
    {
        return $this->post('/predict/growth', $data);
    }

    protected function post(string $path, array $data): array
    {
        $response = Http::timeout(10)->post("{$this->baseUrl}{$path}", $data);

        if ($response->failed()) {
            throw new RequestException($response);
        }

        return $response->json();
    }
}
```

### Registrar el servicio en `AppServiceProvider.php`

```php
use App\Services\TribioMLService;

public function register(): void
{
    $this->app->singleton(TribioMLService::class);
}
```

### Ejemplos de uso en controladores

```php
// ProductController.php
public function mlAnalysis(int $productId)
{
    $product = Product::with('store')->findOrFail($productId);

    $result = $this->ml->predictSales([
        'price'                    => $product->price,
        'discount_pct'             => $product->discount_pct ?? 0,
        'stock'                    => $product->stock ?? 30,
        'images_count'             => $product->images()->count(),
        'description_length'       => strlen($product->description ?? ''),
        'featured'                 => $product->featured ? 1 : 0,
        'payment_settings_enabled' => $product->store->payment_enabled ? 1 : 0,
        'has_whatsapp'             => $product->store->has_whatsapp ? 1 : 0,
        'plan_id'                  => $product->store->plan_id,
        'business_type_slug'       => $product->store->type_slug,
        'business_category_slug'   => $product->store->category_slug,
    ]);

    // $result['probability']      → float 0-1
    // $result['prediction']       → 0 o 1
    // $result['label']            → "venderá ✅" o "no venderá ❌"
    // $result['recommendations']  → array de strings

    return response()->json(['product_id' => $productId, 'ml' => $result]);
}

// CustomerController.php
public function churnRisk(Customer $customer)
{
    $result = $this->ml->predictChurn([
        'days_since_last_order'    => $customer->daysSinceLastOrder(),
        'total_orders_paid'        => $customer->orders()->paid()->count(),
        'avg_order_value'          => $customer->avgOrderValue(),
        'cancellation_rate'        => $customer->cancellationRate(),
        'preferred_payment_method' => $customer->preferred_payment ?? 'cash',
        'preferred_notification'   => $customer->preferred_notification ?? 'whatsapp',
    ]);

    return response()->json($result);
}

// StoreController.php — Reporte completo de una tienda
public function fullReport(Store $store)
{
    return response()->json([
        'store_id' => $store->id,
        'sales'    => $this->ml->predictSales(['price' => $store->avg_product_price, ...]),
        'design'   => $this->ml->predictDesign([
            'payment_settings_enabled' => $store->payment_enabled ? 1 : 0,
            'products_with_image_pct'  => $store->productsWithImagePct(),
            'template_slug'            => $store->template_slug,
            // ...
        ]),
        'growth'   => $this->ml->predictGrowth([
            'posts_count_30d'      => $store->postsLast30Days(),
            'avg_rating'           => $store->avg_rating,
            'has_instagram'        => $store->has_instagram ? 1 : 0,
            // ...
        ]),
    ]);
}
```

---

## 8. Despliegue en producción

### Topología recomendada

```
[Internet]
    │
[Servidor Laravel]  ──red interna──►  [Servidor Python]
  tu-dominio.com                        IP_PRIVADA:8000
                                        uvicorn api:app
```

El servidor Python **no necesita estar expuesto a internet**. Solo debe ser accesible desde el servidor Laravel.

### Con systemd (Linux)

Archivo `/etc/systemd/system/tribio-api.service`:

```ini
[Unit]
Description=TRIBIO IA — ML Prediction API
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/tribio-ml
Environment="TRIBIO_ARTIFACTS_DIR=/opt/tribio-ml/artifacts"
ExecStart=/usr/local/bin/uvicorn api:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable tribio-api
systemctl start tribio-api
systemctl status tribio-api
```

### Con Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY artifacts/ ./artifacts/
COPY api.py .
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

```bash
docker build -t tribio-api .
docker run -d -p 8000:8000 --name tribio-api tribio-api
```

---

## Referencias

- **Dashboard visual**: https://tribio-ml-r5btzwfrf9z8eg3pqnkw2h.streamlit.app/
- **Swagger UI (REST API)**: `https://tonyua-tribio.hf.space/docs`
- **ReDoc**: `https://tonyua-tribio.hf.space/redoc`

---

*TRIBIO IA v1.0.0 — Motor de predicción ML*
