---
name: citation-rules
description: Aplicar reglas de citas y referencias academicas en la tesis DARL. Usar cuando Codex agregue, corrija o revise citas LaTeX, BibTeX, referencias bibliograficas, afirmaciones con soporte, estado del arte o consistencia entre texto y bibliografia.
---

# citation-rules

## Reglas

- Citar toda afirmacion especifica sobre trabajos previos, metricas, algoritmos o benchmarks.
- Verificar que cada clave citada exista en `thesis/referencias.bib` antes de usarla.
- No crear entradas BibTeX con datos incompletos si se puede evitar.
- Marcar preprints o fuentes no revisadas por pares en el texto cuando sea relevante.
- Mantener consistencia entre idioma del texto y formato de cita usado por la plantilla.

## Flujo

1. Identificar afirmaciones que requieren fuente.
2. Buscar clave existente en BibTeX.
3. Insertar cita LaTeX sin romper comandos existentes.
4. Revisar que la referencia respalde exactamente la afirmacion.
5. Evitar sobrecitar una misma frase si una cita al final del enunciado basta.

## Citas frecuentes del proyecto

- TableShift para benchmark tabular bajo distribution shift.
- DISDE y SHIFT para diagnostico de drift.
- CARA y trabajos cost-aware para reentrenamiento.
- Sutton y Barto para fundamentos de RL.
- Schulman et al. para PPO.
- Chen y Guestrin para XGBoost.

## Validacion

Revisar claves BibTeX, ortografia de comandos `\cite{}` y correspondencia entre cita y afirmacion.
