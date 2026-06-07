---
name: literature-review
description: Revisar, sintetizar y conectar literatura academica para tesis-darl. Usar cuando Codex trabaje con papers, notas, brechas de investigacion, comparacion de trabajos, estado del arte, TableShift, drift diagnosis, retraining cost-aware, self-healing pipelines o reinforcement learning aplicado a mantenimiento de modelos.
---

# literature-review

## Alcance

Trabajar con `literature/`, `thesis/referencias.bib` y secciones de revision de literatura en `thesis/`. No agregar PDFs licenciados ni copiar texto extenso de fuentes.

## Flujo

1. Identificar el rol del paper: diagnostico, benchmark, decision de reentrenamiento, remediacion automatica, RL o modelo tabular.
2. Resumir aporte, metodo, evidencia empirica y limitaciones.
3. Conectar explicitamente con la brecha de DARL: seleccionar que etapa actualizar y con que costo.
4. Verificar que toda afirmacion fuerte tenga cita.
5. Mantener citas BibTeX consistentes con `thesis/referencias.bib`.

## Criterio de sintesis

- No listar papers de forma aislada.
- Comparar por pregunta, supuesto, unidad de intervencion, metricas y limitaciones.
- Diferenciar entre detectar drift, explicar degradacion y decidir accion correctiva.
- Marcar preprints o trabajos no revisados por pares cuando corresponda.

## Validacion

Comprobar que las citas existan en BibTeX antes de usarlas. Si falta una referencia, pedir o crear entrada BibTeX solo con datos verificables.
