[🇧🇷 Português](README.md) · [🇪🇸 Español](README.es.md) · 🇬🇧 English (current)

# JHM — Pneumonia Detection Platform (Deep Learning + MLOps)

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-Render-2496ED?logo=docker&logoColor=white)
![CI](https://github.com/hard747/jhm-pneumonia-cnn/actions/workflows/ci-pipeline.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-green)

End-to-end system for automated pneumonia detection in chest X-rays, using
a Convolutional Neural Network (CNN) in PyTorch. Covers the full cycle of
an MLOps project applied to a clinical use case: rigorous model training
and evaluation, a real-time inference API, audit trail and model
versioning in production, real observability (traces and metrics),
complete CI/CD with security scans and semantic versioning, and automatic
cloud deployment.

**Live demo:**
- Frontend: **[jhm-pneumonia-cnn.vercel.app](https://jhm-pneumonia-cnn.vercel.app)** — upload a chest X-ray and get a diagnosis in seconds.
- API: **[jhm-pneumonia-api.onrender.com](https://jhm-pneumonia-api.onrender.com)** ([`/docs`](https://jhm-pneumonia-api.onrender.com/docs) for interactive OpenAPI docs).

> Educational/portfolio use. **Not a medical device** and must not be used
> for real clinical diagnosis — see [Limitations](#limitations-and-responsible-use).

## Table of contents

- [Architecture](#architecture)
- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)
- [Hyperparameter optimization — what worked and what didn't](#hyperparameter-optimization--what-worked-and-what-didnt)
- [API](#api)
- [MLOps: audit trail, versioning, and observability](#mlops-audit-trail-versioning-and-observability)
- [CI/CD](#cicd)
- [Running it locally](#running-it-locally)
- [Repository structure](#repository-structure)
- [Frontend](#frontend)
- [Limitations and responsible use](#limitations-and-responsible-use)
- [Next steps](#next-steps)
- [License and authorship](#license-and-authorship)

## Architecture

```
┌─────────────────┐      HTTPS       ┌──────────────────────┐      inference       ┌─────────────┐
│  React (Vite)    │ ───────────────▶│   FastAPI (Render)    │ ───────────────────▶│ PneumoniaCNN │
│  Vercel          │◀─────────────── │   /predict /health     │◀──────────────────── │  (PyTorch)   │
└─────────────────┘   JSON result    │   /model/info /metrics │     NORMAL/PNEUMONIA └─────────────┘
                                      └──────────┬─────────────┘
                                                  │ SQLAlchemy
                                                  ▼
                                      ┌──────────────────────┐
                                      │  PostgreSQL           │
                                      │  auditorias_diagnostico│
                                      │  model_registry        │
                                      └──────────────────────┘
                                                  │
                                                  │ OTLP (traces + metrics)
                                                  ▼
                                      ┌──────────────────────┐
                                      │  Grafana Cloud        │
                                      └──────────────────────┘
```

Backend and frontend live in the same monorepo (`jhm-pneumonia-cnn/` and
`jhm-pneumonia-frontend/`) but deploy independently: the backend as a
Docker container on Render, the frontend as a static site on Vercel, each
with its own CI/CD pipeline.

## Dataset

- **Source:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — Kaggle, pediatric chest X-rays from Guangzhou Women and Children's Medical Center.
- **Classes:** `NORMAL` / `PNEUMONIA`.
- **Size:** 5,218 training images (1,342 NORMAL / 3,876 PNEUMONIA), 624 test images (234 / 390), 16 official validation images (too small to be used reliably — see the optimization section below).
- **Known caveat about this dataset:** train/ and test/ come from batches/sources with somewhat different image characteristics (a distribution shift documented by other users of the dataset). This directly impacted this project's optimization methodology — see below.

## Model

`PneumoniaCNN` ([`src/model.py`](jhm-pneumonia-cnn/src/model.py)): 3
convolutional blocks (16→32→64 channels, with BatchNorm + ReLU + MaxPool)
followed by two dense layers (128 → 2) with 0.5 dropout. Input: grayscale
X-ray, 128×128. Deliberately simple, fast architecture — the point of this
project is the full MLOps platform around the model, not beating the
state of the art in medical image classification.

## Results

Evaluated in [`src/evaluate.py`](jhm-pneumonia-cnn/src/evaluate.py) on the
624 images in `data/test/`, **never used in any training or
hyperparameter-selection step**:

| Metric | NORMAL | PNEUMONIA |
|---|---|---|
| Precision | 92.6% | 77.7% |
| **Recall** | 53.4% | **97.4%** |
| F1 | 67.8% | 86.5% |

**Overall accuracy: 80.9%** (624 samples). The chosen primary metric is
**PNEUMONIA recall (97.4%)**: in pneumonia screening, a false negative (a
sick patient classified as healthy) is the clinically costliest error —
far more than a false positive, which only means one extra manual review.
The model errs on the safe side: only 10 of the 390 real pneumonia cases
in the test set go undetected.

![Confusion matrix](jhm-pneumonia-cnn/reports/figures/confusion_matrix.png)

## Hyperparameter optimization — what worked and what didn't

This section exists because a well-investigated negative result is still
a real result — hiding it would be less honest than documenting it.

**First attempt:** carve out 15% of `data/train/` as a validation split to
compare 5 configurations (learning rate, class weights, data
augmentation). Result: the best configuration reached **98.2% validation
accuracy** — too good to be true. Measuring that same configuration
against real `data/test/` gave **71.6%** accuracy, worse than the original
configuration (80.9%), with NORMAL recall dropping from 53% to 25%. The
cause: a validation split carved from `train/` doesn't represent the
distribution of `test/` *in this specific dataset* (see the note in the
Dataset section) — the model was overfitting to characteristics specific
to the training batch.

**Second attempt (correct):** validation became a slice of `test/` itself
(40%, "tuning_val"), leaving the other 60% ("holdout_final") completely
untouched until the final decision — never seen during training nor
during configuration selection. The candidate model is compared against
the production model **only** on `holdout_final`
([`src/compare_holdout.py`](jhm-pneumonia-cnn/src/compare_holdout.py)):

| Model | Accuracy | Macro F1 | PNEUMONIA recall | NORMAL recall |
|---|---|---|---|---|
| **Production (original)** | **78.7%** | **0.743** | 96.2% | 49.7% |
| Candidate (augmentation, 10 epochs) | 72.0% | 0.615 | 99.6% | 26.2% |

**Conclusion: none of the tested configurations beat the original model.**
The production configuration remains unchanged. The real value of this
process wasn't "improving a number" — it was identifying a genuine
methodological pitfall (a non-representative validation split), fixing
it, and rigorously confirming the original choice was already solid for
this architecture. Full details in
[`outputs/hyperparameter_search.csv`](jhm-pneumonia-cnn/outputs/hyperparameter_search.csv),
[`outputs/best_config.json`](jhm-pneumonia-cnn/outputs/best_config.json),
and [`outputs/holdout_comparison.json`](jhm-pneumonia-cnn/outputs/holdout_comparison.json).

## API

FastAPI ([`api/main.py`](jhm-pneumonia-cnn/api/main.py)), interactive docs
at [`/docs`](https://jhm-pneumonia-api.onrender.com/docs):

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Basic status and active model version |
| `/health` | GET | Detailed health check (model loaded, database connection) |
| `/model/info` | GET | Active model metadata (MLOps — Model Registry) |
| `/predict` | POST | Takes an image, returns `prediction`, `confidence`, `latency_ms` — rate-limited to 10 requests/minute per IP |
| `/metrics` | GET | Prometheus metrics (requests, latency) |

## MLOps: audit trail, versioning, and observability

- **Audit trail** ([`api/models.py`](jhm-pneumonia-cnn/api/models.py) →
  `AuditoriaDiagnostico`): every prediction made in production is logged to
  Postgres — image, result, confidence, latency, and model version used.
  Full traceability of what the system answered, when, and with which
  model.
- **Model Registry** (`ModelRegistry`): a table versioning deployed models
  (version, accuracy, deploy date, description) — the foundation for
  knowing which model is active and comparing versions over time.
- **Real observability**: OpenTelemetry instruments every `/predict`
  request with distributed traces, exporting both traces and metrics via
  OTLP to **Grafana Cloud** whenever `OTEL_EXPORTER_OTLP_ENDPOINT` is set
  in the environment. Prometheus (`prometheus-fastapi-instrumentator`)
  exposes `/metrics` for additional scraping. Rate limiting (10 req/min
  per IP) protects the inference endpoint from abuse.

## CI/CD

4 GitHub Actions workflows ([`.github/workflows/`](.github/workflows/)):

- **`ci-pipeline.yml`** — on every push/PR to `main`/`dev`: lint + SAST
  (`bandit`) + vulnerable-dependency scan (`safety`) + test suite with
  coverage (`pytest --cov`, reported to Codecov).
- **`deploy-staging.yml`** — a push to `dev` triggers an automatic deploy
  to Render's staging environment.
- **`deploy-prod.yml`** — a push to `main` triggers an automatic deploy to
  production on Render.
- **`release.yml`** — automatic semantic versioning
  ([release-please](https://github.com/googleapis/release-please-action)),
  generating the CHANGELOG from commit messages (Conventional Commits).

Dependabot ([`.github/dependabot.yml`](.github/dependabot.yml)) keeps
Python, npm, and the GitHub Actions themselves updated weekly.

## Running it locally

Backend (requires Docker):

```bash
cd jhm-pneumonia-cnn
cp .env.example .env   # if missing, see docker-compose.yml for the variables
docker compose up
# API available at http://localhost:8000/docs
```

Frontend:

```bash
cd jhm-pneumonia-frontend
npm install
npm run dev
# opens at http://localhost:5173
```

Training/re-evaluating the model (requires the dataset images in `data/`, GPU optional):

```bash
cd jhm-pneumonia-cnn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-train.txt
python -m src.train      # trains from scratch with the original recipe
python -m src.evaluate   # evaluates against data/test/, generates reports/figures/confusion_matrix.png
python -m src.tune       # hyperparameter search (see methodology above)
```

## Repository structure

```
jhm-pneumonia-monorepo/
├── jhm-pneumonia-cnn/            # backend
│   ├── api/                      # FastAPI: main.py, models.py, database.py
│   ├── src/                      # model.py, train.py, evaluate.py, tune.py, compare_holdout.py
│   ├── tests/                    # unit/ + integration/
│   ├── outputs/                  # evaluation_report.json, hyperparameter_search.csv, ...
│   ├── data/                     # dataset (not versioned, see .gitignore)
│   └── Dockerfile
├── jhm-pneumonia-frontend/       # React + Vite
├── .github/workflows/            # CI, staging/prod deploy, release
├── render.yaml                   # backend deploy config
└── LICENSE
```

## Frontend

React 19 + Vite, deployed on Vercel. Upload an X-ray → preview → send to
`/predict` → display the result with a confidence level.

| Initial screen | Result: healthy | Result: pneumonia |
|---|---|---|
| ![Initial screen](jhm-pneumonia-cnn/reports/figures/frontend_screenshot_inicial.png) | ![Healthy result](jhm-pneumonia-cnn/reports/figures/frontend_screenshot_sano.png) | ![Pneumonia result](jhm-pneumonia-cnn/reports/figures/frontend_screenshot_pneumonia.png) |

## Limitations and responsible use

- **Not a medical device** and has not gone through any clinical or
  regulatory validation process. It's an educational/portfolio project
  demonstrating a complete production ML platform.
- The model has a relatively low NORMAL recall (53%) — it tends to
  over-predict PNEUMONIA. This is a deliberate choice (see Results), but
  it means that, in a real scenario, NORMAL results would need additional
  confirmation before being treated as final.
- The dataset has a known distribution bias between train/test (see
  Dataset) — generalization to X-rays from other sources/hospitals isn't
  guaranteed.

## Next steps

- Enrich the dataset with images from multiple sources to reduce the
  distribution shift documented above.
- Explore larger/pre-trained architectures (transfer learning with
  ResNet/DenseNet) as a comparison against the current simple CNN.
- Grafana Cloud dashboards on top of the metrics/traces already being
  exported.

## License and authorship

Distributed under the [MIT](LICENSE) license.

**Author:** Harre Ayma Aranda — sole author of this repository.
