---
name: latex-thesis
description: Editar, revisar y estructurar la tesis LaTeX de tesis-darl en español academico. Usar cuando Codex deba modificar capitulos, secciones, estilo argumentativo, numeracion, referencias cruzadas, figuras, tablas o consistencia formal de la tesis, sin editar archivos generados ni compilar salvo solicitud explicita.
---

# latex-thesis

## Alcance

Trabajar solo sobre fuentes LaTeX versionables dentro de `thesis/`, especialmente `thesis/main.tex`, `thesis/secciones/`, `thesis/encabezados/`, `thesis/images/` y `thesis/tables/`.

No editar manualmente archivos generados en `thesis/build/`. No compilar LaTeX salvo que el usuario lo pida de forma explicita.

## Flujo

1. Revisar `git status --short` antes de mover, borrar o sobrescribir contenido.
2. Leer el archivo `.tex` y el contexto cercano antes de proponer cambios.
3. Mantener redaccion academica en español, con tono preciso y sin inflar alcance.
4. Preservar terminologia del proyecto: DARL, distribution shift, covariate shift, concept drift, pipeline de dos etapas, actualizacion selectiva, POMDP, PPO.
5. Evitar reescrituras amplias si el usuario pide una seccion puntual.
6. Mantener etiquetas, citas y comandos LaTeX existentes salvo que esten claramente rotos.

## Estilo de tesis

- Usar parrafos con tesis clara, evidencia y cierre conceptual.
- Evitar promesas empiricas que aun no esten implementadas o evaluadas.
- Distinguir entre diagnostico de drift, decision de actualizacion y evaluacion de costo.
- Cuando se mencione una contribucion, conectar explicitamente con la brecha de investigacion.
- Para capitulos teoricos, explicar primero el concepto general y luego su rol en DARL.

## Figuras y tablas

- Ubicar figuras generadas en `thesis/figures/generated/` o la ruta vigente del repo si ya existe otra convencion.
- Ubicar tablas generadas en `thesis/tables/generated/` o la ruta vigente del repo si ya existe otra convencion.
- No inventar resultados numericos. Si faltan datos, dejar texto metodologico o marcador explicito.

## Validacion

Revisar sintaxis local de los cambios y que no se hayan tocado builds, PDFs ni salidas generadas. Si no se compila, decirlo en el cierre.
