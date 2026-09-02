# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/).
El versionado semántico automático (release-please) ya gestiona los
releases numerados a partir de los commits; esta sección documenta el
contexto detrás del trabajo de "puesta a punto para portafolio".

## 0.1.0 (2026-09-02)


### Features

* adaptar conexion de base de datos postgres para produccion ([40c0c8d](https://github.com/hard747/jhm-pneumonia-cnn/commit/40c0c8d03cdf39f7f323163b3e7eb3c2e7d43f17))
* add render.yaml with Docker config for jhm-pneumonia-api ([71e34b3](https://github.com/hard747/jhm-pneumonia-cnn/commit/71e34b3198395f4386c4b3720e56de8dc16b2b41))
* avaliacao real do modelo, otimizacao de hiperparametros, observabilidade Grafana Cloud e documentacao completa ([4baf276](https://github.com/hard747/jhm-pneumonia-cnn/commit/4baf2762f55f3c8192073a9b1085ab98fe3e9d18))
* DevOps completo - observabilidad, CI/CD multi-env, seguridad, MLOps, RBAC ([75fb8a7](https://github.com/hard747/jhm-pneumonia-cnn/commit/75fb8a718aaa147117e74248d1a1f9bfd7863676))


### Bug Fixes

* **backend:** agregar numpy a requirements - torchvision lo necesita para transformar imagenes ([303bcab](https://github.com/hard747/jhm-pneumonia-cnn/commit/303bcabdbc0792c1a00efc50eb0b7126fd3df17b))
* **backend:** pinear numpy&lt;2.0.0 e instalar antes de torch - ABI incompatible con numpy 2.x ([275be04](https://github.com/hard747/jhm-pneumonia-cnn/commit/275be04cb1464e537e91a73635d3520a965a93f3))
* **ci-deploy:** corregir incompatibilidades que rompian tests y deploy de Render ([5d99159](https://github.com/hard747/jhm-pneumonia-cnn/commit/5d99159df09ac87a28c6ce36333ef0641c638b94))
* **ci:** aceptar HTTP 202 como respuesta valida del deploy en Render ([de9eca5](https://github.com/hard747/jhm-pneumonia-cnn/commit/de9eca5cb72dbef2fbd4b30068402cec5b2c7500))
* **ci:** corregir 3 causas de fallo en GitHub Actions ([c3f0808](https://github.com/hard747/jhm-pneumonia-cnn/commit/c3f0808ee051849b9b147508961a103df6c5a265))
* **ci:** mejorar diagnóstico del deploy a Render - mostrar HTTP status y body ([c367704](https://github.com/hard747/jhm-pneumonia-cnn/commit/c3677040dedca1092cdba208ea8c91968d709ee8))
* **ci:** pinar httpx&lt;0.27.0 para compatibilidad con Starlette TestClient ([b4c4610](https://github.com/hard747/jhm-pneumonia-cnn/commit/b4c461089e90e1c70ecdf6417d91ffbd1148b9df))
* **ci:** remove unsupported params from release-please-action v4 ([d353cc4](https://github.com/hard747/jhm-pneumonia-cnn/commit/d353cc490a58664821559cab3831d9027afd3672))
* **frontend:** cambiar async/await a .then() para evitar bug de React 19 con handlers async ([9efe25d](https://github.com/hard747/jhm-pneumonia-cnn/commit/9efe25d377d3ea8a3963a238422bb2156db1967a))
* **frontend:** URL produccion Render como fallback y mover setLoading dentro del try-catch ([8bedc4b](https://github.com/hard747/jhm-pneumonia-cnn/commit/8bedc4b561740cd1435740a8541be1fb9d2f0a7d))
* pin prometheus-fastapi-instrumentator==5.9.1 para Python 3.8 ([7ca854f](https://github.com/hard747/jhm-pneumonia-cnn/commit/7ca854ffb4aafc40e2e2903ebc72f74a01bd8817))
* requirements.txt encoding UTF-8 sin BOM, pip podia leerlo ([2fd9a16](https://github.com/hard747/jhm-pneumonia-cnn/commit/2fd9a16a60b22dbab55b7e0f2b3630d1f10c344b))
* simplify apt deps, remove opencv system libs not used in API ([ff221b0](https://github.com/hard747/jhm-pneumonia-cnn/commit/ff221b0a01f60643e0bed4476e0dddf1e50c6709))
* UTF-8 requirements, CPU torch para Render, puerto dinamico ([55bf995](https://github.com/hard747/jhm-pneumonia-cnn/commit/55bf99586497fe2006f637dcba8f40d48e4e92ac))


### Documentation

* actualizar screenshots del frontend en el README ([2d24f8b](https://github.com/hard747/jhm-pneumonia-cnn/commit/2d24f8b8c0bf937efeeebb825986425625ead05c))

## [Unreleased]

### Added — avaliação e otimização real do modelo
- `src/evaluate.py`: reporte de evaluación real contra `data/test/`
  (accuracy, precision/recall/F1 por clase, matriz de confusión),
  guardado en `outputs/evaluation_report.json` y
  `reports/figures/confusion_matrix.png`.
- `src/tune.py`: búsqueda real de hiperparámetros (learning rate, class
  weights, data augmentation), con selección de configuración basada en
  una porción del propio `data/test/` (no de `data/train/`) tras
  descubrir un train/test distribution shift en el dataset público — ver
  README para el detalle completo del hallazgo.
- `src/compare_holdout.py`: comparación justa entre el modelo candidato y
  el de producción, sobre la porción de test nunca usada en la búsqueda.
- Corregido un valor de accuracy hardcodeado e inexacto (`0.974`) en el
  endpoint `/model/info`, reemplazado por el valor real medido (`0.8093`).

### Added — observabilidad real (Grafana Cloud)
- `api/main.py`: exportador OTLP de métricas (además del ya existente de
  trazas), ambos configurables vía `OTEL_EXPORTER_OTLP_ENDPOINT` +
  `OTEL_EXPORTER_OTLP_HEADERS`, compatibles con Grafana Cloud.
- `render.yaml`: variables de entorno declaradas para la conexión OTLP.

### Added — documentación de portafolio
- READMEs completos en portugués (idioma de entrada), español e inglés:
  arquitectura, dataset, resultados, metodología de optimización,
  endpoints de la API, CI/CD, cómo correrlo localmente.
