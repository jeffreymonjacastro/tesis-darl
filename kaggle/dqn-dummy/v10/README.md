# DDQN v10: búsqueda bayesiana del reward

Implementación preparada desde v9. Estado observado tras el envío: **RUNNING**, versión 1.

[Notebook en Kaggle](https://www.kaggle.com/code/jeffreyamc/darl-ddqn-v10-bayesian-reward). Se comprobó coincidencia exacta del código local y remoto, metadata privada T4 e internet deshabilitado. No se esperó su finalización ni se descargaron resultados.

Kaggle derivó el slug del título; la metadata local quedó alineada con el identificador real sin lanzar otra versión.

## Protocolo

- Cuatro agentes independientes, 1000 episodios cada uno; semilla 42.
- 300 camas, 40 días, warmup siete días y 33 decisiones por episodio.
- Mismas 52 entradas, red, inyectores, replays, particiones, referencias y calendarios de v9; preparación privada de v8 idéntica a la usada por v9, sin cargar pesos DDQN.
- `r = alpha*AUC_post + beta*AP_post + eta*(delta_auc+delta_ap) - lambda*C`, con `C=seconds/reference_A4`.
- Alpha, beta, eta y lambda independientes en [0.1,1]. Una sola eta para ambas diferencias.
- Dos puntos iniciales y dos propuestas adaptativas mediante proceso gaussiano Matérn y Expected Improvement, sin dependencias nuevas ni internet.

El criterio aprobado por el usuario es **media del retorno nativo de episodios 901–1000 menos media de episodios 1–100**. BO y elección global maximizan esa misma diferencia; empate exacto conserva el primer trial. Los pesos cambian su escala: el resultado no demuestra por sí solo mayor rendimiento absoluto ni clínico. AUC, AP, costo y niveles finales se muestran por separado.

Dentro de cada trial se conserva la selección de pesos de v9: último checkpoint de la mejor ventana de tres validaciones consecutivas. La configuración ganadora se congela antes de evaluar test, contra A1–A4, Random y heurística.

## Outputs

Solo los detalles por decisión y predicciones de la ganadora quedan en el resultado final. Como excepción necesaria para reproducir las comparaciones, se conservan un CSV compacto de cuatro trials y las series de retornos por episodio de todos ellos; no se conservan CSV de observaciones, acciones ni predicciones de agentes perdedores.

```text
outputs/
├── README.md
├── figures/                    20 PNG previstos
├── predictions/                test de la ganadora y baselines
├── models/                     checkpoint seleccionado
├── checkpoints/                búsqueda, snapshot y último estado de entrenamiento
├── analysis/
│   ├── training/
│   ├── validation/
│   ├── test/
│   ├── comparisons/            resumen compacto de búsqueda
│   └── plot_data/              series, suavizados, bandas y varianzas
└── metadata/
    ├── configuration/
    ├── preparation/
    └── checks/
```

Los trials activos y el mejor provisional se conservan temporalmente bajo checkpoints/trials/ para no perder sus historiales ni volver a medir tiempos. Se eliminan tras publicar la ganadora completa. No se duplican tablas diarias equivalentes a tablas de decisiones. Las auditorías de etiquetas se consolidan en un CSV por paciente/schedule.

Modo `plots`: reconstruye figuras desde los CSV organizados, sin dataset, inferencia ni entrenamiento. Las figuras 14–16 comparan curvas nativas, parámetros y criterio; 17–20 muestran acciones por episodio/checkpoint, promedios/proporciones, secuencia diaria y AUC con políticas fijas. Las bandas son variación temporal de una semilla.

## Ejecución y límites

Solo FULL en Kaggle CUDA/T4; XGBoost CPU con dos hilos dentro de Kaggle. Imagen Docker fijada como v9, kernel privado e internet deshabilitado. JSON UTF-8 con escapes Unicode para evitar la recodificación de Windows CLI.

Se guarda estado atómico de la búsqueda, sugerencia activa, ensayos completos, pesos, optimizador, replay, RNG y offsets CSV del trial. El límite interno es 11 horas, con timeout remoto de 12 horas. Cuatro ejecuciones similares a v9 suman unas 10.1 horas antes de evaluación/exportación: si se llega al límite, el estado será partial, sin ganador final ni reducción de episodios. Una continuación requiere montar el output parcial y la misma preparación v8; no se lanza automáticamente.

La validación local es JSON/AST y revisión de fuente. Los contratos numéricos, ejecución y renderizado real se verifican dentro del FULL remoto. No se ejecutan modelos ni búsqueda GP localmente. La calibración heredada no cumplía sus criterios en v9 y se reportará separadamente de la integridad.

El usuario pidió **solo lanzar**: no esperar a que termine ni descargar resultados hasta su aviso.

## Resultados descargados — 2026-11-12

FULL completo: cuatro trials, 4000 episodios y 132000 decisiones; 5 h 10 min. Se verificaron los 106 archivos, 105 hashes del manifiesto y 97 checks remotos. [Informe de hallazgos](outputs/full-20261112/local-analysis/README.md). Ganó el trial 3 por aumento del reward, principalmente debido al menor costo; la política seleccionada usa A1 en 129/132 decisiones y no supera A1 fija en test. La calibración continúa False. No se ejecutaron modelos ni entrenamiento local, ni se relanzó Kaggle.
