---
name: results-to-latex
description: Convertir resultados experimentales de DARL en tablas, figuras y fragmentos LaTeX para la tesis. Usar cuando Codex tome metricas desde outputs, genere tablas comparativas, exporte figuras o redacte interpretaciones empiricas basadas en resultados validados.
---

# results-to-latex

## Alcance

Leer resultados desde `outputs/`. Escribir artefactos finales en `thesis/figures/generated/` o `thesis/tables/generated/` cuando existan esas rutas; si el repo usa rutas equivalentes, seguir la convencion vigente.

No inventar numeros. No mover resultados sin revisar `git status --short`.

## Flujo

1. Leer resultados fuente y confirmar columnas, unidades y seeds.
2. Validar que las metricas representen el experimento indicado.
3. Generar tablas o figuras reproducibles desde `outputs/`.
4. Exportar artefactos aptos para LaTeX.
5. Redactar interpretacion breve: hallazgo, evidencia, implicacion para DARL y limitacion.

## Reglas de tablas

- Incluir dataset, modelo, tipo de drift, severidad, accion, AUC antes/despues, delta AUC, tiempo y RAM cuando aplique.
- Ordenar por pregunta experimental, no por conveniencia del archivo.
- Usar precision numerica consistente.

## Reglas de figuras

- Etiquetar ejes con unidades.
- Evitar colores ambiguos si se comparan acciones.
- Mantener leyendas legibles para insercion en tesis.

## Validacion

Reportar ruta del archivo fuente, ruta del artefacto exportado y cualquier fila descartada o supuesto aplicado.
