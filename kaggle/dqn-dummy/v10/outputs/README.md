# Outputs DDQN v10

Ejecución privada `jeffreyamc/darl-ddqn-v10-bayesian-reward`, versión 1. Estado Kaggle e interno: complete. Tesla T4, 18579 s (5 h 10 min), cuatro trials FULL de 1000 episodios y 132000 decisiones totales.

## Descarga y verificación — 2026-11-12

- `full-20261112/ddqn-v10/`: 106 archivos originales, con la estructura figures/predictions/models/checkpoints/analysis/metadata acordada.
- 20 PNG y 28 CSV de predicciones de test.
- Verificados tamaños y SHA-256 de los 105 archivos del manifiesto (el propio manifiesto es el archivo 106), sin modificar originales.
- 97 comprobaciones remotas de integridad aprobadas; contratos numéricos y bayesianos aprobados. Se reconstruyó el criterio de los cuatro trials desde 4000 filas de curvas, y la aritmética del reward de la ganadora.
- `full-20261112/local-analysis/README.md`: informe independiente; CSV de componentes iniciales/finales, promedios de test y verificación. Solo procesamiento local de CSV, sin modelos ni entrenamiento.
- `status-20261112/`: resumen y log descargados inicialmente. `submitted-v1/` y `submission.json`: recibo original de envío.

## Hallazgo principal

Ganó el trial 3 por crecimiento del reward (+6.285419), con alpha=0.557965, beta=0.454248, eta=0.892371 y lambda=0.749632. Checkpoint 930 seleccionado mediante validación dentro del trial.

La penalización media por costo cae de 7.862211 a 0.780633 entre los primeros y últimos 100 episodios. Los componentes predictivos ponderados y G disminuyen; el reward aumenta principalmente por reducir costo. No se interpreta como mejor adaptación predictiva al drift.

Test DDQN: A1=129, A2=3, A3=A4=0. Retorno medio 10.693341, AUC diaria 0.548868, AP diaria 0.048351. A1 fija: retorno 10.886441, AUC 0.551354 y AP 0.048998. A4 tiene mayor AUC/AP pero reward negativo con estos pesos. Los retornos usan la fórmula ganadora; no comparar sus magnitudes directamente con v9.

La calibración heredada sigue False. Cuatro puntos en cuatro dimensiones y una semilla no prueban un óptimo global. No se dispone de test de los trials perdedores. No se lanzó otra ejecución.
