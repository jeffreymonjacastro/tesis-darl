---
name: formula-explanation
description: Redactar y revisar formulas de la tesis DARL con el patron obligatorio formula, Donde:, explicacion de simbolos y explicacion conceptual. Usar cuando Codex agregue o corrija ecuaciones de drift, metricas, POMDP, PPO, recompensas, costos, AUC, PSI, KS, C2ST o severidad.
---

# formula-explanation

## Patron obligatorio

Usar siempre esta secuencia:

1. Formula en entorno LaTeX adecuado.
2. Bloque `Donde:`.
3. Definicion de cada simbolo usado en la formula.
4. Explicacion conceptual en español academico.

## Reglas

- Definir todos los simbolos nuevos, incluidos subindices, superindices y operadores.
- Mantener consistencia con notacion ya usada en la tesis.
- No introducir una notacion alternativa si existe una vigente.
- Separar senales no supervisadas de covariate shift, senales supervisadas de concept drift y metricas de performance drift.
- Evitar derivaciones largas si la tesis solo requiere interpretacion.

## Conceptos DARL frecuentes

- `P(X)` para distribucion marginal de covariables.
- `P(Y|X)` para relacion condicional asociada a concept drift.
- `PSI_t`, `D_{KS,t}`, `C2ST_t`, `S_X(t)` para monitoreo de covariate shift.
- `\Delta AUC_t` y `\Delta \ell_t` para degradacion supervisada.
- `o_t` para vector de observacion del agente.
- `A` para acciones: diferir, actualizar preprocesamiento, actualizar modelo, reentrenar pipeline completo.

## Validacion

Comprobar que cada formula tenga `Donde:` y explicacion conceptual. Revisar que no falten simbolos por definir.
