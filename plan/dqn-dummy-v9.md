# DDQN v9: horizonte observable y monitoreo por variable

> Ejecución: seguir este plan por tareas; validación estructural local y contratos numéricos exclusivamente en Kaggle FULL.

**Objetivo:** crear v9 desde v8, ejecutar 1000 episodios en Kaggle T4, descargar los resultados y comparar retorno, acciones y estimaciones Q con el FULL de v8.

**Arquitectura:** conservar el experimento de 300 camas, 40 días, siete días iniciales y 33 decisiones. Añadir el horizonte a la historia de observaciones y permitir elegir KS/PSI agregados o por signo vital. Incorporar diagnósticos TD sin modificar el objetivo de aprendizaje.

**Tecnologías:** notebook Python, PyTorch DDQN CUDA, XGBoost CPU con dos hilos y Kaggle CLI.

**Especificación:** objetivo adjunto `ebfe7d7e-0331-4783-8b64-42758728ca55/goal-objective.md` y confirmación del usuario: conservar **+G**.

## Restricciones

- Semilla 42, 1000 episodios y 33 000 decisiones; ejecución experimental solo en Kaggle GPU FULL.
- No modificar v8 ni sus outputs. No ejecutar simulaciones, inferencia ni entrenamiento local.
- Reward intacto: `AUC_post + AP_post + (AUC_post-AUC_pre) + (AP_post-AP_pre) - 0.025 C`.
- Compartir preparación, particiones, modelos iniciales, referencias KS/PSI, calibración, referencia de costo A4 y calendario con el FULL de v8; comprobar identidad antes de reutilizar sus artefactos privados.
- Mantener máscaras, siete días históricos etiquetados, retraso de un día, red 128–128–4, gamma 0.99, Huber, target cada 50 gradientes, batch 64, buffer 50 000 y calendarios LR/epsilon.
- Un agente FULL en modo `per_vital` por defecto. La alternativa `aggregate` queda implementada y documentada; no se declarará evaluada sin una ejecución independiente.
- Selección solo con validación; el test no decide el checkpoint. La comparación con v8 mide el cambio conjunto, sin atribuir causalidad individual a cada componente.

## Tarea 1: observaciones y contratos

- [x] Crear `kaggle/dqn-dummy/v9/input/main.ipynb` y metadata privada desde v8.
- [x] Implementar modos `aggregate` (5 valores diarios) y `per_vital` (7 KS, 7 PSI, caída AUC, AP y validez: 17 valores diarios).
- [x] Concatenar tres días y un escalar `h=(40-day+1)/33`: 16 o 52 entradas. Comprobar h=1 en día 8, 1/33 en día 40 y 0 en estado final día 41.
- [x] Mantener normalizaciones fijas y orden de vitales; exportar valores crudos, transformados y validez del monitoreo. Rechazar métricas por variable inválidas sin sustituirlas silenciosamente por ausencia de drift.
- [x] Añadir contratos numéricos de ambos modos, días predictivos inválidos, causalidad y fronteras temporales. Ejecutarlos dentro del notebook FULL.

## Tarea 2: evidencia Q/TD y reanudación

- [x] Exportar Q por decisión también en entrenamiento, horizonte, `terminated` y `truncated`.
- [x] Guardar diagnósticos de minibatch por bloques de actualizaciones, separados entre terminales/no terminales: número de muestras, Q, objetivo TD, error firmado/absoluto, RMSE y Huber.
- [x] Verificar que cada objetivo terminal sea su recompensa y que el estado siguiente tenga horizonte cero exactamente en transiciones terminales.
- [x] Persistir acumuladores de diagnóstico y offsets CSV para continuar sin perder ni duplicar muestras.
- [x] Comprobar que el número de muestras diagnosticadas sea `64 × updates`.

## Tarea 3: envío, ejecución y descarga

- [x] Validar JSON/AST, metadata, fórmula sin cambios, identidad del baseline y ausencia de outputs en las celdas.
- [x] Revisar el cambio completo con un revisor independiente antes del envío.
- [x] Enviar `jeffreyamc/darl-dqn-dummy-v9`, privado, T4, internet deshabilitado, FULL; registrar recibo y código enviado.
- [x] Monitorizar el mismo job hasta estado terminal. Si falla, diagnosticar primero; no relanzar por un timeout de observación.
- [x] Descargar todos los outputs en un directorio nuevo bajo `v9/outputs/` y comprobar estado interno, conteos, archivos y contratos.

## Tarea 4: comparación y entrega

- [x] Comparar v8/v9 por checkpoint y escenario sobre las mismas trayectorias: retorno, AUC, AP, costo, acciones, variación temporal y brecha mejor ventana/última ventana de tres validaciones.
- [x] Comparar Q de observación inicial, separación mejor/segunda acción, retorno descontado observado y error terminal. No confundir Q con reward diario ni exigir coincidencia con cada realización estocástica.
- [x] Regenerar figuras desde CSV, revisar PNG y documentar resultados en README de v9 y de outputs.
- [x] Reportar por separado estado remoto, integridad, calibración, rendimiento y límites de una sola semilla. No exigir mejora, curvas monótonas ni uso de las cuatro acciones para declarar correcta la ejecución.

## Decisiones de implementación

- Se trabaja en la rama existente `feature/ddqn-experiments`, con v9 aislada en una carpeta nueva; no se hacen commits ni push sin solicitud.
- Se reutiliza la preparación de v8, sin reutilizar pesos ni buffer DDQN: evita cambiar pacientes, calibración o denominador del costo al comparar.
- El modo por variable exige métricas finitas para los siete vitales. Su ausencia queda registrada y produce error explícito; los bloques iniciales artificiales siguen marcados como inválidos.
- Un futuro experimento de flujo continuo o cambios de régimen dentro del episodio se conserva como trabajo posterior y no forma parte de este cambio controlado.

## Resultado y cierre — 2026-11-05

- Kernel `jeffreyamc/darl-dqn-dummy-v9`, versión 1: estado remoto e interno `complete`, FULL, Tesla T4, 9127 segundos. Una sola ejecución: 1000 episodios y 33 000 decisiones, 28 evaluaciones en test.
- Descarga completa de 213 archivos (1 577 522 138 bytes) en `kaggle/dqn-dummy/v9/outputs/full-20261105/ddqn-v9/`. Manifiesto comprobado, hashes registrados y snapshot verificado por CRC sin deserializar modelos. Los cortes de red se resolvieron con descargas reanudables, sin relanzar Kaggle.
- 97 comprobaciones remotas de integridad, 16 contratos numéricos y 109 comprobaciones independientes aprobadas. Preparación idéntica a v8; etiquetas/pacientes iguales en las 28 tablas de test, con 237 600 registros cada una. Objetivos TD terminales sin bootstrap y muestras diagnosticadas `32 937 × 64 = 2 107 968`.
- Selección solo en validación: checkpoint 740. Mejor ventana de tres validaciones: 21.915520 (v8: 21.684487); última ventana: 21.817092 (v8: 21.192905); brecha: 0.098428 (v8: 0.491582). Discrepancia Q terminal media de las últimas diez validaciones: 0.404241 (v8: 59.524710).
- Test: retorno 21.483952 (v8: 21.448836), AUC diaria 0.594090 (0.591759), AP diaria 0.057728 (0.056843). La mejora predictiva es pequeña y no uniforme: retorno covariate +0.4531, combined −0.3955. Acciones: A1=6, A2=14, A3=96, A4=16.
- Trece figuras regeneradas desde CSV y tres comparativas, con revisión visual. El paquete remoto queda intacto. Informe: `kaggle/dqn-dummy/v9/outputs/comparison-v8-v9/README.md`.
- Calibración `False` en v8 y v9. El resultado respalda Q más coherentes y menor retroceso de validación en esta realización, sin demostrar superioridad general ni aislar horizonte frente a KS/PSI por vital. El modo `aggregate` tiene contratos aprobados, pero no un entrenamiento independiente. No se realizó una interrupción/reanudación deliberada; se verificaron implementación y snapshot final.
