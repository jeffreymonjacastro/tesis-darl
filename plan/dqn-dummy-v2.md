# DARL DQN Dummy v2: cambios respecto a v1

## 1. Motivo

En v1 el DQN eligió A1 en 57 de 60 decisiones de prueba y A4 en 3. No fue un
error: el oracle inmediato de `action_audit.csv` también prefería A1 en 52 de 60
decisiones. Con un único cambio persistente, una sola actualización adapta el
pipeline y después deferir es lo mejor. Además, el costo `0,1 × segundos`
equivalía a unas 0,003 unidades de AUC, por lo que A4 dominaba a A2 y A3.

La versión 2 modifica el **entorno** para que A2 y A3 sean decisiones con
sentido. No se editan resultados ni se añaden bonos por acción.

## 2. Cambios

### A. Cambio por etapas

- Tres etapas acumulativas. En prueba y validación empiezan el día 5 y cada etapa
  dura 4 días.
- Covariate, etapa k: `m_ref + scale · (x − m_ref) + k · offset · iqr_ref · dirección`.
- Concept, etapa k: la etiqueta pasa a `y_concept{k}`, generada con la misma regla
  que `y_dummy` pero con signos invertidos en los pesos: HR; HR y Resp; Resp.
- Combined: el concepto empieza media etapa después que la medición, de modo que
  ambos cambios se alternan.
- Monitoreo KS/PSI relativo a la referencia del último ajuste del transformador
  (A2/A4 la reemplazan), para que cada nueva etapa de medición sea observable.
- La calibración con B-train exige que un segundo A2 recupere la etapa 2 y que A3
  recupere la etapa 1 de concepto.

### C. Costo comparable

`reward = AUC_día_siguiente − 0,05 · segundos / mediana_segundos_A4`

La unidad se mide en B-train antes de calibrar y entrenar, y queda congelada.
Este cambio modifica la definición de reward de v1 y se declara como desviación
del plan original.

### D. Exploración

- Epsilon decae una vez por episodio con factor 0,985 y mínimo 0,05; en v1
  decaía por actualización y llegaba a 0,01 alrededor del episodio 65.
- Ciclo de entrenamiento: seis de cada siete episodios con cambio.
- El inicio (días 3–7) y la duración de las etapas (3–5 días) se sortean en
  entrenamiento con una semilla fija.

## 3. Sin cambios

Datos, particiones, replay, pipeline, arquitectura DQN, baselines, oracle
inmediato, figuras 1–8 y gates de v1. Se añade el gate `dqn_used_a2_and_a3`.

## 4. Kaggle

Kernel privado `jeffreyamc/darl-dqn-dummy-v2`; entrada en
`kaggle/dqn-dummy/v2/input/` y salidas en `kaggle/dqn-dummy/v2/outputs/`.
