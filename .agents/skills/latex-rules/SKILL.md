---
name: latex-rules
description: "Edita y valida la tesis DARL en LaTeX: redacción académica, estructura, citas, fórmulas y la incorporación trazable de resultados experimentales en tablas y figuras. Usar para cambios que llegan a thesis/; no para desarrollar experimentos sin una entrega para la tesis."
---

# Escritura LaTeX para la tesis DARL

Usar esta skill al crear, revisar o integrar contenido de la tesis DARL. Mantener una cadena de evidencia: resultados reproducibles en `outputs/` antes de tablas, figuras o afirmaciones empíricas en `thesis/`.

## Flujo común

1. Revisar `git status --short`, el archivo fuente y el contexto antes de editar o mover contenido.
2. Seguir la estructura y convenciones reales del repositorio; consultar [estructura de la tesis](references/thesis-structure.md) para cambios de organización, inclusiones, figuras, tablas o compilación.
3. Redactar en español académico: afirmación clara, evidencia verificable e implicación. No presentar diseño metodológico, diagnósticos internos o resultados preliminares como hallazgos concluyentes.
4. Mantener etiquetas, citas y comandos existentes salvo que estén rotos. Usar etiquetas únicas y referencias cruzadas (`\label`, `\ref`, `\eqref`) en lugar de numeración escrita a mano.
5. Tras modificar fuentes LaTeX, compilar desde `thesis/` con `latexmk -pdf -outdir=build main.tex`. Corregir errores propios y reportar los errores externos que impidan compilar.

## Rutas especializadas

- Para bibliografía, afirmaciones sustentadas o ecuaciones, leer [citas y fórmulas](references/citations-and-formulas.md).
- Para diseñar evidencia experimental que se entregará a la tesis, validar resultados o exportar tablas/figuras, leer [evidencia y artefactos](references/evidence-and-artifacts.md).
- Evitar reescrituras amplias ante una edición localizada. Antes de borrar o mover artefactos existentes, confirmar su uso mediante referencias e `git status`.

## Límites

Esta skill gobierna la escritura y la evidencia que llega a `thesis/`. Para cambios de implementación que no produzcan un artefacto o una afirmación para la tesis, trabajar con las reglas de Python o del experimento aplicables, sin forzar una edición LaTeX.
