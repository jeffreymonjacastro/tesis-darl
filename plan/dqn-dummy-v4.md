# DARL DQN Dummy v4: cambio gradual

## Sin cambios respecto a v3

Datos, particiones, replay, pipeline y acciones A1–A4. También la observación
(KS/PSI respecto al último ajuste de features, AUC, tres días) y la recompensa
`AUC_día_siguiente − 0,05 · s / mediana_s(A4)`. El agente es el mismo: Double DQN,
pérdida Huber, gamma 0,99, Adam 5e-4 → 5e-5, epsilon 0,99 por episodio con
mínimo 0,05, 500–1000 episodios y selección por media móvil de tres validaciones.
Los gates son los mismos.

## Cambio: severidad gradual

En v2 y v3 cada etapa aparecía de golpe, lo que producía caídas de AUC de un día.
En v4 la severidad crece cada día desde el inicio de la rampa y nunca se detiene.

- Covariate: `L(d) = rate_cov · (d − inicio + 1)` IQR, sin tope; escala
  `1 + 0,4 · min(1, L/3)`.
- Concept: `u(d) = rate_con · (d − inicio + 1)`, limitado a 3. Los pesos se
  interpolan linealmente entre W → HR invertido → HR y Resp → Resp. El umbral es
  la mediana del score de A-train con esos pesos, y se aplica la misma inversión
  del 5%.
- Combined: ambas rampas; la de concepto empieza 2 días después.

Una actualización no detiene el cambio: la degradación vuelve a acumularse. El
agente debe decidir cuándo la pérdida acumulada justifica el costo.

## Calibración con B-train

Rejillas: `rate_cov ∈ {0,25; 0,35; 0,5}` IQR/día y `rate_con ∈ {0,10; 0,15; 0,20}`
unidades/día. Se elige la velocidad más lenta que cumple tres condiciones:

1. Caída de AUC menor que 0,05 el primer día.
2. Caída mayor que 0,15 tras seis días sin actualizar.
3. Recuperación mayor que 0,10 al actuar ese día (A2 para covariate, A3 para concept).

En entrenamiento se sortean el inicio (días 3–7) y un factor de velocidad de
0,75, 1 o 1,25.

## Figuras

Las figuras 1–8 son las de v3 adaptadas: la figura 2 muestra la severidad diaria
y las líneas verticales marcan el inicio de cada rampa. Se añade la figura 9,
AUC diaria del DQN frente a A1.

Kernel: `jeffreyamc/darl-dqn-dummy-v4`.
