---
title: Tribio ML API
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
short_description: Motor de predicción ML para e-commerce — REST API FastAPI
---

# TRIBIO IA — REST API

Motor de predicción ML de 4 módulos para negocios de e-commerce.

## Endpoints

| Endpoint | Descripción |
|----------|-------------|
| `GET /health` | Estado del servicio |
| `POST /predict/sales` | Predicción de ventas (7 días) |
| `POST /predict/churn` | Riesgo de churn (30 días) |
| `POST /predict/design` | Conversión según diseño (30 días) |
| `POST /predict/growth` | Crecimiento por contenido (7 días) |

## Documentación interactiva

- **Swagger UI**: https://tonyua-tribio.hf.space/docs
- **ReDoc**: https://tonyua-tribio.hf.space/redoc
- **Dashboard visual**: https://tribio-ml-r5btzwfrf9z8eg3pqnkw2h.streamlit.app/

## Uso rápido

```bash
# Health check
curl https://tonyua-tribio.hf.space/health

# Predicción de ventas
curl -X POST https://tonyua-tribio.hf.space/predict/sales \
  -H "Content-Type: application/json" \
  -d '{"price": 55.0, "images_count": 4, "payment_settings_enabled": 1}'
```

## Respuesta estándar

```json
{
  "probability": 0.8214,
  "prediction": 1,
  "label": "venderá ✅",
  "threshold": 0.42,
  "recommendations": ["..."]
}
```
