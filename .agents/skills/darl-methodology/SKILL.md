---
name: darl-methodology
description: Diseñar, revisar y redactar la metodologia de DARL a partir de la tesis. Cubre POMDP, PPO, observaciones de drift, acciones de actualizacion selectiva, recompensas, datasets TableShift, simulacion de episodios y protocolo experimental. Usar cuando Codex trabaje en Capitulo III, entorno RL, decision policy, baselines o alineacion entre teoria e implementacion.
---

# darl-methodology

## Alcance

Usar esta skill para alinear la metodologia escrita y la implementacion experimental de DARL. La tesis define DARL como un agente PPO que decide entre cuatro acciones de mantenimiento para pipelines tabulares de dos etapas bajo drift.

## Elementos obligatorios

- Problema: actualizacion selectiva de pipeline tabular de dos etapas.
- Estado oculto: distribuciones reales `P_t(X)`, `P_t(Y|X)` y degradacion real.
- Observacion: vector con `PSI_t`, `D_{KS,t}`, `C2ST_t`, `S_X(t)`, `\Delta AUC_t` y `\Delta \ell_t`.
- Acciones: diferir, actualizar preprocesamiento, actualizar modelo predictivo, reentrenar pipeline completo.
- Recompensa: combinar recuperacion de AUC-ROC y costo computacional relativo.
- Agente: PPO con arquitectura Actor-Critico.
- Evaluacion: comparar politica aprendida contra baseline empirico o tabla de decision.

## Flujo para Capitulo III

1. Definir objetivo experimental y pregunta que responde.
2. Describir datasets, splits y generacion de drift.
3. Describir pipeline base y acciones selectivas.
4. Formalizar POMDP con observaciones, acciones, transiciones y recompensa.
5. Describir entrenamiento PPO y configuracion reproducible.
6. Definir metricas: AUC-ROC, F1, delta AUC, tiempo y RAM.
7. Describir baselines y protocolo de comparacion.
8. Separar claramente diseño metodologico de resultados.

## Flujo para codigo

1. Implementar entorno Gymnasium con observacion numerica estable.
2. Encapsular acciones de actualizacion como funciones medibles.
3. Registrar costos y recuperacion por episodio.
4. Guardar resultados en `outputs/` antes de exportar a tesis.
5. Mantener semillas y configuraciones versionables.

## Validacion

Comprobar que cada elemento metodologico tenga contraparte en codigo o plan experimental. Si una parte aun no existe, escribirla como decision de diseño, no como resultado ejecutado.
