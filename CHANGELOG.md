# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/).
El versionado semántico automático (release-please) ya gestiona los
releases numerados a partir de los commits; esta sección documenta el
contexto detrás del trabajo de "puesta a punto para portafolio".

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
