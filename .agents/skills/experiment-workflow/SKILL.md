---
name: experiment-workflow
description: Diseñar y ejecutar experimentos reproducibles de DARL en el repo tesis-darl. Usar cuando Codex implemente o revise cargas TableShift, inyeccion de drift, pipelines de dos etapas, monitoreo Evidently/River, metricas, runners, configuraciones, semillas y outputs experimentales.
---

# experiment-workflow

## Alcance

Trabajar en `code/`, `data/`, `outputs/`, `scripts/` y configuraciones experimentales. No versionar datasets, modelos entrenados, logs pesados, caches ni resultados temporales.

## Flujo

1. Revisar estructura vigente del repo antes de crear modulos.
2. Fijar semillas en `42` para Python, NumPy, modelos y simulaciones cuando aplique.
3. Implementar modulos pequeños antes de experimentos grandes.
4. Ejecutar sanity checks por modulo: shapes, balance de clases, salida minima o prueba pequeña.
5. Escribir resultados primero en `outputs/`.
6. Exportar a tesis solo despues de validar resultados.

## Componentes DARL

- Data loading: TableShift para datasets tabulares definidos por la tesis.
- Pipeline: preprocesamiento y modelo predictivo refitteables por separado.
- Drift injector: covariate, concept y both con severidad controlada.
- Monitoring: PSI, KS, C2ST, Evidently y River/ADWIN cuando aplique.
- Metrics: AUC-ROC, F1, delta AUC, tiempo de reentrenamiento y RAM.
- Runner: CSV de resultados con columnas reproducibles.

## Validacion

Antes de cerrar, reportar comando ejecutado, resultado del sanity check y archivos generados. Si no se ejecuto por costo o dependencias, decirlo claramente.
