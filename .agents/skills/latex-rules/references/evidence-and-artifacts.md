# Evidencia experimental, tablas y figuras

Usar esta referencia cuando una solicitud conecte experimentos DARL con una afirmación, tabla o figura de la tesis.

## Experimentos reproducibles orientados a la tesis

- Revisar el contexto y la estructura vigente antes de crear módulos o configuraciones. Mantener semillas en `42` para Python, NumPy, modelos y simulaciones cuando aplique.
- Implementar y validar componentes pequeños antes de un experimento grande. Registrar un sanity check proporcional: shapes, balance de clases, salida mínima o prueba pequeña.
- Para DARL, verificar que carga de datos, inyección de drift, pipeline de dos etapas, monitoreo, acciones y métricas correspondan a la pregunta experimental. No describir una capacidad no ejecutada como resultado.
- Conservar el comando, configuración, semilla, resultado del sanity check y rutas generadas. Si el experimento no se ejecutó por coste o dependencia, declararlo claramente.

## Del resultado a la tesis

1. Leer el resultado fuente en `outputs/` y comprobar columnas, unidades, semillas y alcance.
2. Validar que la métrica responda la pregunta experimental. Para DARL, priorizar recuperación real de AUC, calidad de acción/política, retorno comparativo y costo (tiempo/RAM); no usar frecuencia de acciones ni diagnósticos PPO como prueba de corrección de la política.
3. Generar de forma reproducible la tabla o figura en `outputs/tables/` o `outputs/figures/`.
4. Exportar únicamente el artefacto final curado a `thesis/tables/generated/` o `thesis/figures/generated/`.
5. Incorporarlo desde la sección correspondiente y redactar una interpretación breve: hallazgo, evidencia, implicación para DARL y limitación.

## Tablas

- Ordenar por pregunta experimental, no por conveniencia del archivo.
- Incluir según aplique: dataset, modelo, tipo y severidad de drift, acción, AUC antes/después, `\Delta AUC`, tiempo y RAM.
- Mantener precisión numérica y unidades consistentes; no redondear de una forma que cambie la conclusión.
- Cada fila debe ser rastreable al resultado fuente. Declarar cualquier fila descartada o supuesto aplicado.

## Figuras

- Etiquetar ejes y unidades, usar leyendas legibles y distinguir acciones sin depender solo del color.
- Mostrar condiciones relevantes: dataset, escenario de drift, severidad o horizonte, cuando sean necesarias para interpretar la comparación.
- No usar figuras de recompensa PPO como evidencia suficiente de recuperación predictiva: describirlas como diagnóstico de entrenamiento si ese es su alcance.
