[🇧🇷 Português](README.md) · 🇪🇸 Español (actual) · [🇬🇧 English](README.en.md)

# JHM — Plataforma de Detección de Neumonía (Deep Learning + MLOps)

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-Render-2496ED?logo=docker&logoColor=white)
![CI](https://github.com/hard747/jhm-pneumonia-cnn/actions/workflows/ci-pipeline.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-green)

Sistema end-to-end de detección automática de neumonía en radiografías de
tórax, usando una Red Neuronal Convolucional (CNN) en PyTorch. Cubre todo
el ciclo de un proyecto de MLOps aplicado a un caso de uso clínico:
entrenamiento y evaluación rigurosa del modelo, API de inferencia en
tiempo real, auditoría y versionamiento de modelos en producción,
observabilidad real (trazas y métricas), CI/CD completo con escaneos de
seguridad y versionamiento semántico, y deploy automático en la nube.

**Demo en vivo:**
- Frontend: **[jhm-pneumonia-cnn.vercel.app](https://jhm-pneumonia-cnn.vercel.app)** — subí una radiografía y recibí el diagnóstico en segundos.
- API: **[jhm-pneumonia-api.onrender.com](https://jhm-pneumonia-api.onrender.com)** ([`/docs`](https://jhm-pneumonia-api.onrender.com/docs) para la documentación interactiva OpenAPI).

> Uso educacional/portafolio. **No es un dispositivo médico** y no debe
> usarse para diagnóstico clínico real — ver [Limitaciones](#limitaciones-y-uso-responsable).

## Índice

- [Arquitectura](#arquitectura)
- [Dataset](#dataset)
- [Modelo](#modelo)
- [Resultados](#resultados)
- [Optimización de hiperparámetros — qué funcionó y qué no](#optimización-de-hiperparámetros--qué-funcionó-y-qué-no)
- [API](#api)
- [MLOps: auditoría, versionamiento y observabilidad](#mlops-auditoría-versionamiento-y-observabilidad)
- [CI/CD](#cicd)
- [Cómo correrlo localmente](#cómo-correrlo-localmente)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Frontend](#frontend)
- [Limitaciones y uso responsable](#limitaciones-y-uso-responsable)
- [Próximos pasos](#próximos-pasos)
- [Licencia y autoría](#licencia-y-autoría)

## Arquitectura

```
┌─────────────────┐      HTTPS       ┌──────────────────────┐      inferencia      ┌─────────────┐
│  React (Vite)    │ ───────────────▶│   FastAPI (Render)    │ ───────────────────▶│ PneumoniaCNN │
│  Vercel          │◀─────────────── │   /predict /health     │◀──────────────────── │  (PyTorch)   │
└─────────────────┘   resultado JSON │   /model/info /metrics │     NORMAL/PNEUMONIA └─────────────┘
                                      └──────────┬─────────────┘
                                                  │ SQLAlchemy
                                                  ▼
                                      ┌──────────────────────┐
                                      │  PostgreSQL           │
                                      │  auditorias_diagnostico│
                                      │  model_registry        │
                                      └──────────────────────┘
                                                  │
                                                  │ OTLP (trazas + métricas)
                                                  ▼
                                      ┌──────────────────────┐
                                      │  Grafana Cloud        │
                                      └──────────────────────┘
```

Backend y frontend viven en el mismo monorepo (`jhm-pneumonia-cnn/` y
`jhm-pneumonia-frontend/`) pero se despliegan de forma independiente: el
backend como container Docker en Render, el frontend como sitio estático
en Vercel, cada uno con su propio pipeline de CI/CD.

## Dataset

- **Fuente:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — Kaggle, radiografías de tórax pediátricas del Guangzhou Women and Children's Medical Center.
- **Clases:** `NORMAL` / `PNEUMONIA`.
- **Tamaño:** 5.218 imágenes de entrenamiento (1.342 NORMAL / 3.876 PNEUMONIA), 624 de test (234 / 390), 16 de validación oficial (demasiado pequeña para usarse de forma confiable — ver la sección de optimización más abajo).
- **Nota conocida sobre este dataset:** train/ y test/ vienen de lotes/fuentes con características de imagen algo distintas (un *distribution shift* documentado por otros usuarios del dataset). Esto tuvo impacto directo en la metodología de optimización de este proyecto — ver más abajo.

## Modelo

`PneumoniaCNN` ([`src/model.py`](jhm-pneumonia-cnn/src/model.py)): 3 bloques
convolucionales (16→32→64 canales, con BatchNorm + ReLU + MaxPool)
seguidos de dos capas densas (128 → 2) con dropout 0.5. Entrada: radiografía
en escala de grises, 128×128. Arquitectura simple y rápida a propósito —
el objetivo del proyecto es la plataforma MLOps completa alrededor del
modelo, no batir el estado del arte en clasificación de imágenes médicas.

## Resultados

Evaluado en [`src/evaluate.py`](jhm-pneumonia-cnn/src/evaluate.py) sobre
las 624 imágenes de `data/test/`, **nunca usadas en ninguna etapa de
entrenamiento ni de selección de hiperparámetros**:

| Métrica | NORMAL | PNEUMONIA |
|---|---|---|
| Precision | 92,6% | 77,7% |
| **Recall** | 53,4% | **97,4%** |
| F1 | 67,8% | 86,5% |

**Accuracy general: 80,9%** (624 muestras). La métrica primaria elegida es
el **recall de PNEUMONIA (97,4%)**: en el cribado de neumonía, un falso
negativo (paciente enfermo clasificado como sano) es el error clínicamente
más caro — mucho más que un falso positivo, que solo implica una revisión
manual extra. El modelo se equivoca del lado seguro: solo 10 de los 390
casos reales de neumonía en el test set pasan desapercibidos.

![Matriz de confusión](jhm-pneumonia-cnn/reports/figures/confusion_matrix.png)

## Optimización de hiperparámetros — qué funcionó y qué no

Esta sección existe porque un resultado negativo, bien investigado,
también es un resultado real — y ocultarlo sería menos honesto que
documentarlo.

**Primer intento:** separar 15% de `data/train/` como validación para
comparar 5 configuraciones (learning rate, class weights, data
augmentation). Resultado: la mejor configuración alcanzó **98,2% de
accuracy en validación** — demasiado bueno para ser cierto. Al medir esa
misma configuración contra `data/test/` real, la accuracy fue de **71,6%**,
peor que la configuración original (80,9%), con el recall de NORMAL
cayendo de 53% a 25%. La causa: una validación recortada de `train/` no
representa la distribución de `test/` *en este dataset específicamente*
(ver nota en la sección Dataset) — el modelo estaba sobreajustando a
características específicas del lote de entrenamiento.

**Segundo intento (correcto):** la validación pasó a ser una porción del
propio `test/` (40%, "tuning_val"), dejando el otro 60% ("holdout_final")
completamente aislado hasta la decisión final — nunca visto durante el
entrenamiento ni durante la elección de configuración. Se compara el
modelo candidato contra el modelo en producción **solo** en
`holdout_final` ([`src/compare_holdout.py`](jhm-pneumonia-cnn/src/compare_holdout.py)):

| Modelo | Accuracy | F1 macro | Recall PNEUMONIA | Recall NORMAL |
|---|---|---|---|---|
| **Producción (original)** | **78,7%** | **0,743** | 96,2% | 49,7% |
| Candidato (augmentation, 10 épocas) | 72,0% | 0,615 | 99,6% | 26,2% |

**Conclusión: ninguna configuración probada superó al modelo original.**
La configuración de producción se mantiene sin cambios. El valor real de
este proceso no fue "mejorar un número" — fue identificar una trampa
metodológica real (validación no representativa), corregirla, y confirmar
con rigor que la elección original ya era sólida para esta arquitectura.
Detalles completos en [`outputs/hyperparameter_search.csv`](jhm-pneumonia-cnn/outputs/hyperparameter_search.csv),
[`outputs/best_config.json`](jhm-pneumonia-cnn/outputs/best_config.json) y
[`outputs/holdout_comparison.json`](jhm-pneumonia-cnn/outputs/holdout_comparison.json).

## API

FastAPI ([`api/main.py`](jhm-pneumonia-cnn/api/main.py)), documentación
interactiva en [`/docs`](https://jhm-pneumonia-api.onrender.com/docs):

| Endpoint | Método | Descripción |
|---|---|---|
| `/` | GET | Estado básico y versión del modelo activo |
| `/health` | GET | Health check detallado (modelo cargado, conexión a la base) |
| `/model/info` | GET | Metadatos del modelo activo (MLOps — Model Registry) |
| `/predict` | POST | Recibe una imagen, devuelve `prediction`, `confidence`, `latency_ms` — limitado a 10 solicitudes/minuto por IP |
| `/metrics` | GET | Métricas Prometheus (requests, latencia) |

## MLOps: auditoría, versionamiento y observabilidad

- **Auditoría** ([`api/models.py`](jhm-pneumonia-cnn/api/models.py) →
  `AuditoriaDiagnostico`): cada predicción hecha en producción queda
  registrada en Postgres — imagen, resultado, confianza, latencia y
  versión del modelo usado. Trazabilidad completa de qué respondió el
  sistema, cuándo y con qué modelo.
- **Model Registry** (`ModelRegistry`): tabla que versiona los modelos
  desplegados (versión, accuracy, fecha de deploy, descripción) — la base
  para saber qué modelo está activo y comparar versiones a lo largo del
  tiempo.
- **Observabilidad real**: OpenTelemetry instrumenta cada solicitud
  (`/predict`) con trazas distribuidas, y exporta trazas + métricas vía
  OTLP a **Grafana Cloud** cuando `OTEL_EXPORTER_OTLP_ENDPOINT` está
  configurado en el entorno. Prometheus (`prometheus-fastapi-instrumentator`)
  expone `/metrics` para scraping adicional. Rate limiting (10 req/min por
  IP) protege el endpoint de inferencia contra abuso.

## CI/CD

4 workflows en GitHub Actions ([`.github/workflows/`](.github/workflows/)):

- **`ci-pipeline.yml`** — en cada push/PR a `main`/`dev`: lint + SAST
  (`bandit`) + escaneo de dependencias vulnerables (`safety`) + suite de
  tests con cobertura (`pytest --cov`, reportada en Codecov).
- **`deploy-staging.yml`** — push a `dev` dispara deploy automático al
  ambiente de staging de Render.
- **`deploy-prod.yml`** — push a `main` dispara deploy automático a
  producción en Render.
- **`release.yml`** — versionamiento semántico automático
  ([release-please](https://github.com/googleapis/release-please-action)),
  genera el CHANGELOG a partir de los mensajes de commit (Conventional
  Commits).

Dependabot ([`.github/dependabot.yml`](.github/dependabot.yml)) mantiene
las dependencias de Python, npm y las propias GitHub Actions actualizadas
semanalmente.

## Cómo correrlo localmente

Backend (requiere Docker):

```bash
cd jhm-pneumonia-cnn
cp .env.example .env   # si no existe, ver docker-compose.yml para las variables
docker compose up
# API disponible en http://localhost:8000/docs
```

Frontend:

```bash
cd jhm-pneumonia-frontend
npm install
npm run dev
# abre en http://localhost:5173
```

Entrenar/reevaluar el modelo (requiere las imágenes del dataset en `data/`, GPU opcional):

```bash
cd jhm-pneumonia-cnn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-train.txt
python -m src.train      # entrena desde cero con la receta original
python -m src.evaluate   # evalúa contra data/test/, genera reports/figures/confusion_matrix.png
python -m src.tune       # búsqueda de hiperparámetros (ver metodología arriba)
```

## Estructura del repositorio

```
jhm-pneumonia-monorepo/
├── jhm-pneumonia-cnn/            # backend
│   ├── api/                      # FastAPI: main.py, models.py, database.py
│   ├── src/                      # model.py, train.py, evaluate.py, tune.py, compare_holdout.py
│   ├── tests/                    # unit/ + integration/
│   ├── outputs/                  # evaluation_report.json, hyperparameter_search.csv, ...
│   ├── data/                     # dataset (no versionado, ver .gitignore)
│   └── Dockerfile
├── jhm-pneumonia-frontend/       # React + Vite
├── .github/workflows/            # CI, deploy staging/prod, release
├── render.yaml                   # config de deploy del backend
└── LICENSE
```

## Frontend

React 19 + Vite, deploy en Vercel. Subida de radiografía → preview → envío
a `/predict` → visualización del resultado con nivel de confianza.

| Pantalla inicial | Resultado: sano | Resultado: neumonía |
|---|---|---|
| ![Pantalla inicial](jhm-pneumonia-cnn/reports/figures/frontend_screenshot_inicial.png) | ![Resultado sano](jhm-pneumonia-cnn/reports/figures/frontend_screenshot_sano.png) | ![Resultado neumonía](jhm-pneumonia-cnn/reports/figures/frontend_screenshot_pneumonia.png) |

## Limitaciones y uso responsable

- **No es un dispositivo médico** y no pasó por ningún proceso de
  validación clínica o regulatoria. Es un proyecto educacional/de
  portafolio que demuestra una plataforma de ML en producción de punta a
  punta.
- El modelo tiene recall de NORMAL relativamente bajo (53%) — tiende a
  sobre-predecir PNEUMONIA. Es una elección deliberada (ver Resultados),
  pero significa que, en un escenario real, resultados NORMAL necesitarían
  confirmación adicional antes de tratarse como definitivos.
- El dataset tiene un sesgo de distribución conocido entre train/test (ver
  Dataset) — la generalización a radiografías de otras fuentes/hospitales
  no está garantizada.

## Próximos pasos

- Enriquecer el dataset con imágenes de múltiples fuentes para reducir el
  distribution shift documentado arriba.
- Explorar arquitecturas más grandes/preentrenadas (transfer learning con
  ResNet/DenseNet) como comparación contra la CNN simple actual.
- Dashboards en Grafana Cloud sobre las métricas/trazas ya exportadas.

## Licencia y autoría

Distribuido bajo licencia [MIT](LICENSE).

**Autor:** Harre Ayma Aranda — único autor de este repositorio.
