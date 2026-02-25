# TRIBIO IA — Guía de integración para motores de IA

> Este documento está diseñado para ser leído por un agente de IA (LLM) que necesite
> consumir la API de predicción de Tribio. Incluye instrucciones en lenguaje natural,
> esquemas de datos y ejemplos de llamadas HTTP listos para usar.

---

## ¿Qué es TRIBIO IA?

TRIBIO IA es un motor de predicción de machine learning para negocios de e-commerce.
Expone una REST API con 4 módulos de predicción. Cada módulo responde una pregunta
de negocio específica enviando datos del negocio como JSON y recibe una probabilidad,
una predicción binaria y recomendaciones accionables.

**Dashboard visual** (solo lectura, no es la API):
```
https://tribio-ml-r5btzwfrf9z8eg3pqnkw2h.streamlit.app/
```

**REST API** (para integración programática):
```
http://TU_SERVIDOR:8000
```
> Reemplaza `TU_SERVIDOR` con la IP o dominio donde está desplegado el servicio FastAPI.

---

## Reglas de uso para el agente de IA

1. **Siempre llamar primero a `/health`** para verificar que los modelos están cargados.
2. **Todos los endpoints son POST** (excepto `/health` que es GET).
3. **Content-Type**: siempre `application/json`.
4. **Solo enviar los campos que tienes disponibles**. Todos los campos opcionales tienen defaults razonables.
5. **Interpretar la respuesta**: usar `prediction` (0/1) para la decisión binaria y `recommendations` para acciones.
6. **En caso de error 503**: el modelo ML no está cargado. Reportar el problema al usuario sin reintentar indefinidamente.
7. **En caso de error 422**: falta un campo requerido o el tipo es incorrecto. Revisar los campos obligatorios del endpoint.

---

## Endpoints disponibles

### 1. Health Check

```
GET http://TU_SERVIDOR:8000/health
```

Úsalo para verificar disponibilidad. Si `models_loaded.m1` (o cualquier módulo) es `false`, ese módulo no está disponible.

**Respuesta esperada:**
```json
{
  "status": "ok",
  "models_loaded": { "m1": true, "m2": true, "m3": true, "m4": true },
  "version": "1.0.0"
}
```

---

### 2. Módulo 1 — Predicción de ventas de producto

**Pregunta:** ¿Este producto venderá en los próximos 7 días?

```
POST http://TU_SERVIDOR:8000/predict/sales
```

**Cuándo usarlo:** Cuando tengas datos de un producto (precio, imágenes, stock, etc.)
y quieras saber si es probable que venda en la próxima semana.

**Campos requeridos:**
- `price` (número decimal) — precio del producto

**Campos opcionales con sus defaults:**
```json
{
  "discount_pct": 0.0,
  "stock": 30,
  "images_count": 3,
  "description_length": 400,
  "featured": 0,
  "payment_settings_enabled": 1,
  "has_whatsapp": 1,
  "plan_id": "basic",
  "business_type_slug": "store",
  "business_category_slug": "retail"
}
```

**Ejemplo de llamada completa:**
```json
POST /predict/sales
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

**Respuesta:**
```json
{
  "probability": 0.8214,
  "prediction": 1,
  "label": "venderá ✅",
  "threshold": 0.42,
  "recommendations": ["Ficha sólida: prueba A/B de precio y CTA para subir ticket y conversión."]
}
```

**Cómo interpretar:**
- `prediction == 1` → el modelo predice que el producto SÍ venderá
- `prediction == 0` → el modelo predice que el producto NO venderá
- `probability` → certeza del modelo (cuanto más alto, más seguro)
- `recommendations` → acciones específicas para mejorar ese producto

---

### 3. Módulo 2 — Predicción de churn de cliente

**Pregunta:** ¿Este cliente dejará de comprar en los próximos 30 días?

```
POST http://TU_SERVIDOR:8000/predict/churn
```

**Cuándo usarlo:** Cuando tengas datos de comportamiento de un cliente y quieras
identificar si está en riesgo de abandono para activar retención preventiva.

**Campos requeridos:**
- `days_since_last_order` (entero) — días desde su último pedido
- `total_orders_paid` (entero) — cuántos pedidos ha pagado en total
- `avg_order_value` (decimal) — valor promedio de sus pedidos

**Campos opcionales con sus defaults:**
```json
{
  "cancellation_rate": 0.05,
  "profile_visits_count": 10,
  "link_click_to_order_ratio": 1.2,
  "preferred_payment_method": "cash",
  "preferred_notification": "whatsapp"
}
```

**Ejemplo de llamada:**
```json
POST /predict/churn
{
  "days_since_last_order": 25,
  "total_orders_paid": 2,
  "avg_order_value": 70.0,
  "cancellation_rate": 0.05,
  "preferred_payment_method": "cash",
  "preferred_notification": "whatsapp"
}
```

**Respuesta:**
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

**Cómo interpretar:**
- `prediction == 1` → cliente EN RIESGO de churn → activar estrategia de retención
- `prediction == 0` → cliente ESTABLE → mantener estrategia actual
- `recommendations` → acciones de retención personalizadas según los datos del cliente

---

### 4. Módulo 3 — Predicción de conversión según diseño/config de tienda

**Pregunta:** ¿La configuración actual de esta tienda generará alta conversión en 30 días?

```
POST http://TU_SERVIDOR:8000/predict/design
```

**Cuándo usarlo:** Cuando tengas datos de configuración de una tienda y quieras
evaluar si su setup actual es óptimo para convertir visitas en ventas.

**Todos los campos son opcionales.** Envía los que tengas disponibles.

**Campos con sus defaults:**
```json
{
  "payment_settings_enabled": 1,
  "hero_has_cta": 1,
  "hero_slides_count": 2,
  "navigation_menu_items_count": 5,
  "has_custom_logo": 1,
  "has_cover_image": 1,
  "products_with_image_pct": 0.80,
  "products_with_description_pct": 0.60,
  "total_products_active": 60,
  "products_with_discount_pct": 0.25,
  "avg_images_per_product": 2.5,
  "plan_id": "basic",
  "template_slug": "Minimal",
  "business_type_slug": "store"
}
```

**Valores aceptados para campos de tipo string:**
- `plan_id`: `free` | `basic` | `pro` | `enterprise`
- `template_slug`: `Minimal` | `Classic` | `NikeStyle` | `DarkLuxury` | `Valentine` | `Academy`
- `business_type_slug`: `store` | `appointments` | `restaurant`

**Respuesta:**
```json
{
  "probability": 0.7541,
  "prediction": 1,
  "label": "alta conversión ✅",
  "threshold": 0.45,
  "recommendations": ["Config sólida: ahora prueba A/B con template, topbar y banners."]
}
```

**Cómo interpretar:**
- `prediction == 1` → ALTA probabilidad de conversión
- `prediction == 0` → conversión MEDIA o BAJA → revisar recomendaciones
- `recommendations` → mejoras concretas de diseño y configuración

---

### 5. Módulo 4 — Predicción de crecimiento por contenido

**Pregunta:** ¿El negocio crecerá la próxima semana basado en su estrategia de contenido?

```
POST http://TU_SERVIDOR:8000/predict/growth
```

**Cuándo usarlo:** Cuando tengas métricas de contenido y redes sociales del negocio
y quieras predecir si la estrategia actual generará crecimiento en los próximos 7 días.

**Todos los campos son opcionales.** Envía los que tengas disponibles.

**Campos con sus defaults:**
```json
{
  "posts_count_30d": 10,
  "stories_count_30d": 20,
  "pct_posts_with_video": 0.50,
  "days_between_posts": 3.0,
  "avg_views_per_post": 1500.0,
  "avg_likes_per_post": 60.0,
  "avg_comments_per_post": 6.0,
  "avg_rating": 4.5,
  "pct_1_2_star": 0.05,
  "reviews_with_photo_pct": 0.20,
  "post_engagement_rate": 0.04,
  "business_type_slug": "store",
  "has_instagram": 1,
  "has_tiktok": 1,
  "has_facebook": 1,
  "has_whatsapp": 1
}
```

**Respuesta:**
```json
{
  "probability": 0.7102,
  "prediction": 1,
  "label": "crecerá ✅",
  "threshold": 0.27,
  "recommendations": ["Estrategia sólida: haz A/B test de formato (reels vs carrusel) y horario de publicación."]
}
```

**Cómo interpretar:**
- `prediction == 1` → el negocio CRECERÁ según la estrategia actual
- `prediction == 0` → el negocio estará ESTABLE o BAJARÁ → cambiar estrategia
- `recommendations` → ajustes específicos de contenido y redes sociales

---

## Estructura de respuesta unificada

Todos los endpoints de predicción devuelven exactamente este esquema:

```typescript
{
  probability: number;      // 0.0 a 1.0 — score del modelo ML
  prediction: 0 | 1;        // 1 = positivo, 0 = negativo
  label: string;            // texto amigable del resultado
  threshold: number;        // umbral de decisión del modelo
  recommendations: string[]; // lista de acciones accionables (1 o más)
}
```

---

## Flujo de trabajo recomendado para el agente

```
1. Llamar GET /health
   → Si status != "ok" o modelo no cargado: reportar error y detener

2. Recolectar los datos del objeto de negocio (producto/cliente/tienda)

3. Llamar el endpoint apropiado con los datos disponibles
   → Solo enviar los campos que tienes; los demás usarán sus defaults

4. Leer la respuesta:
   a. prediction == 1 → resultado positivo
   b. prediction == 0 → resultado negativo, revisar recommendations
   c. probability > 0.7 → alta confianza
   d. probability entre 0.4-0.7 → confianza moderada, considerar contexto adicional
   e. probability < 0.4 → baja confianza, el modelo tiene incertidumbre

5. Presentar al usuario:
   - El resultado (label)
   - La probabilidad (como porcentaje: probability * 100)
   - Las recomendaciones
```

---

## Ejemplo de sistema prompt para agente de IA

Si quieres que un agente LLM use esta API automáticamente, puedes incluir este bloque en su system prompt:

```
Tienes acceso a la API de TRIBIO IA en http://TU_SERVIDOR:8000.
Esta API tiene 4 módulos de predicción ML para negocios de e-commerce:

- POST /predict/sales → predice si un producto venderá en 7 días (requiere: price)
- POST /predict/churn → predice riesgo de churn de un cliente en 30 días
  (requiere: days_since_last_order, total_orders_paid, avg_order_value)
- POST /predict/design → predice alta conversión según config de tienda (todos opcionales)
- POST /predict/growth → predice crecimiento según contenido/engagement (todos opcionales)

Todas las respuestas tienen: probability (0-1), prediction (0|1), label, threshold, recommendations[].

Cuando el usuario te pida analizar un producto, cliente o tienda:
1. Identifica qué módulo corresponde
2. Extrae los datos necesarios de la conversación o contexto
3. Llama al endpoint apropiado
4. Presenta el resultado en lenguaje natural con las recomendaciones

Si no tienes todos los campos opcionales, envía solo los que tengas disponibles.
```

---

## Códigos de error a manejar

| Código | Acción recomendada |
|--------|--------------------|
| `200` | Procesar respuesta normalmente |
| `422` | Identificar campo faltante o con tipo incorrecto. Revisar `detail[].msg` |
| `503` | Modelo no disponible. Informar al usuario que el servicio ML no está listo |
| `500` | Error del servidor. Reintentar una vez, luego escalar al equipo técnico |
| Connection error | El servidor Python no está disponible. Verificar despliegue |

---

## Glosario de campos

| Campo | Descripción |
|-------|-------------|
| `probability` | Número entre 0 y 1. Cuanto más cercano a 1, más seguro está el modelo |
| `prediction` | 1 = el evento SÍ ocurrirá, 0 = el evento NO ocurrirá |
| `threshold` | El corte que usa el modelo para decidir entre 0 y 1 (ya está optimizado, no modificar) |
| `recommendations` | Acciones específicas que el modelo sugiere basado en los datos de entrada |
| `plan_id` | Nivel de suscripción del negocio en la plataforma Tribio |
| `business_type_slug` | Tipo de negocio: `store` (tienda), `appointments` (citas), `restaurant` |
| `featured` | Si el producto aparece destacado en la tienda (1 = sí, 0 = no) |
| `payment_settings_enabled` | Si la tienda acepta pagos online (1 = sí, 0 = solo efectivo/manual) |
| `has_whatsapp` | Si el negocio tiene WhatsApp de atención activo |
| `cancellation_rate` | Proporción de pedidos del cliente que fueron cancelados (0 = ninguno, 1 = todos) |
| `post_engagement_rate` | Interacciones totales divididas entre alcance del post |
| `pct_posts_with_video` | Proporción de publicaciones que incluyen video (0 = ninguna, 1 = todas) |

---

*TRIBIO IA v1.0.0 — Documentación para integración con agentes de IA*
