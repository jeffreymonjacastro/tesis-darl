# Citas, referencias y fórmulas

## Citas y bibliografía

1. Identificar las afirmaciones específicas sobre trabajos previos, algoritmos, métricas, benchmarks o resultados comparativos que necesitan respaldo.
2. Comprobar que cada clave exista en `thesis/referencias.bib` antes de escribir `\cite{...}`.
3. Confirmar que la fuente respalda exactamente la afirmación, no solo el tema general. Marcar preprints o fuentes no revisadas por pares cuando sea relevante.
4. Mantener las citas IEEE/natbib vigentes y el idioma español del texto. Una cita al final de un enunciado suele ser suficiente; evitar sobrecitar.

Referencias frecuentes en DARL incluyen TableShift, DISDE, SHIFT, enfoques cost-aware, Sutton y Barto (RL), Schulman et al. (PPO) y Chen y Guestrin (XGBoost). Su mención no reemplaza la verificación de la clave ni de la afirmación concreta.

## Fórmulas

Al agregar o corregir una fórmula, usar esta secuencia:

1. Ecuación en el entorno LaTeX adecuado y con etiqueta si será referenciada.
2. Párrafo o bloque **Donde:**.
3. Definición de cada símbolo nuevo, índice, superíndice u operador.
4. Explicación conceptual en español académico: qué mide, cómo se interpreta y para qué se usa en DARL.

No introducir notación alternativa si ya existe una notación vigente. Distinguir las señales no supervisadas de covariate shift (`P(X)`, PSI, KS, C2ST), las señales supervisadas de concept drift (`P(Y\mid X)`) y la degradación de desempeño (`\Delta AUC`, pérdida). Para el entorno de decisión, mantener la relación consistente entre observación `o_t`, acciones y recompensa.

Evitar derivaciones largas si la sección solo exige una interpretación. Antes de cerrar, comprobar que ningún símbolo haya quedado sin definir y que las referencias cruzadas de ecuaciones compilen.
