# Outputs v11 B

## Descarga verificada — 2026-11-12

Kernel jeffreyamc/darl-ddqn-dummy-v11-b versión2: Kaggle COMPLETE, estado interno PARTIAL. Cierre controlado a39603 segundos (11h), razón training_episode. Siete trials completos y trial8 episodio896; 7896/10000 episodios, faltan2104. No hay ganadora final ni test.

`run-20261112/ddqn-v11-b/` conserva el bundle original: 295 archivos, 841880031 bytes. Se verificaron tamaños de transferencia y se registraron hashes SHA256 locales. No hay manifiesto remoto final porque el run es parcial. CRC de snapshots trial6/trial8, último estado de entrenamiento6 y pesos seleccionados aprobados; estado BO legible con7 observaciones y trial8 activo. No se ejecutaron modelos ni una reanudación local.

El bundle completo permite preparar una continuación; no basta el selected.pt aislado. La compatibilidad de un destino distinto debe verificarse antes de reanudar; esta descarga no demuestra una continuación ejecutada.

Informe: [hallazgos parciales](analysis/comparisons/partial-20261112/README.md). Evidencia de descarga/checkpoints: analysis/comparisons/partial-20261112/download_verification.json.

## Hallazgo

Solo trial6 cumple40% en validación, checkpoint520: A1=51, A2=40, A3=41, A4=0 (132 decisiones). Alpha0.9672430610 beta0.1053078723 eta0.9215749166 lambda0.1006156445. Crecimiento de reward0.890029; AUC diaria de validación0.589172, AP0.053081. No son resultados de test.

Trial4 crece8.139341 pero concentra85.61% y queda excluido. La referencia v10 concentra97.73% en validación y tampoco cumple. El40% se aplica agregado en cuatro escenarios; no fuerza cuatro acciones ni balance por escenario. Faltan completar8/9/10 antes de seleccionar una ganadora definitiva y consultar test.

Los recibos originales en metadata/ documentan el envío y diagnóstico inicial. No se relanzó ninguna ejecución.
