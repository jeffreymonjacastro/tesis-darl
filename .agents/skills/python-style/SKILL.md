---
name: python-style
description: Mantener codigo Python modular, reproducible y documentado para el paquete experimental DARL. Usar cuando Codex edite modulos, clases, runners, notebooks, tests, configuracion de dependencias o estilo de codigo en code/, scripts/ y experimentos.
---

# python-style

## Principios

- Preferir modulos pequeños con responsabilidades claras.
- Usar docstrings breves en clases y metodos publicos.
- Fijar semillas en `42` cuando haya aleatoriedad.
- Evitar logica experimental escondida en notebooks si debe ser reutilizable.
- Usar APIs sklearn-compatible cuando el componente sea parte del pipeline.

## Estructura esperada

- `code/src/darl/data/` para carga y preparacion.
- `code/src/darl/drift/` para inyeccion y diagnostico de drift.
- `code/src/darl/pipeline/` para pipelines de dos etapas.
- `code/src/darl/actions/` para estrategias de actualizacion.
- `code/src/darl/rl/` para entorno, agente y politica.
- `code/src/darl/evaluation/` para metricas y comparaciones.
- `code/src/darl/visualization/` para graficos.

## Reglas de implementacion

- Separar configuracion, entrenamiento, evaluacion y escritura de resultados.
- No descargar datasets ni ejecutar experimentos pesados sin necesidad.
- Medir costo computacional con funciones explicitas.
- Retornar estructuras tabulares o dataclasses antes de escribir CSV.
- Evitar dependencias globales si el proyecto usa entorno local.

## Validacion

Ejecutar pruebas o sanity checks focalizados. Reportar si no se pudieron correr por dependencias, red o costo.
