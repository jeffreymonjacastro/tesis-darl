# DDQN v7 — barrido independiente AUC/AP/costo

## Alcance

Base: notebook local v6. v5 y v6 se conservan. Solo entrenamiento remoto
Kaggle T4, modo FULL: 200 camas, 30 días, 1000 episodios por agente,
27 decisiones por episodio. Nueve agentes; 9000 episodios y 243 000 decisiones.

Reward: `alpha*AUC + beta*AP - lambda*(s_t/s_A4)`.
Alpha y beta independientes en {0.5,0.75,1}; lambda fijo en 0.025.
Estado continuo de 15 entradas y particiones/protocolo XGBoost de v6.
La selección usa exclusivamente B-validation y utilidad común
`0.5*AUC + 0.5*AP - 0.025*C`; no el retorno nativo entre rewards distintos.

## Decisiones de implementación

- Mantener IQR covariate y conversión temporal de negativos de v6.
  Para positivos, solo adelanto de hasta 24 h: el pedido dice adelantar.
  No eliminar eventos ni cambiar etiquetas originales; solo nuevos ingresos.
- Una sesión inicial v7 y continuaciones v8/v9 si hace falta. Mismo barrido,
  sin reiniciar episodios ni reducir 1000. Persistir preparación, estado completo
  del agente y CSV al terminar cada checkpoint de diez episodios. El límite
  temporal produce `partial`, nunca `complete`.
- Continuaciones montan outputs de la sesión previa como kernel source.
  Referencia A4, modelo inicial, trayectorias y calendario se reutilizan.
- XGBoost CPU con dos hilos dentro de Kaggle; red DDQN en GPU obligatoria.
- Predicciones float64 y lectura round-trip para evitar que redondeo CSV cambie
  clasificaciones exactamente en el umbral F1.

## Entregables y evidencia

CSV largos con alpha/beta/lambda, fases, episodios, escenarios y días.
Ocho familias de figuras heredadas, identificando los tres pesos; suavizado100
y ±1 DE, varianza exportada. Tablas/figuras temporales adicionales de v6.
Modo plots exclusivamente CSV. Selección congelada antes de B-test.
Verificación remota de rewards, historia15, causalidad, máscaras, conservación
de etiquetas, métricas exportadas y 9000 episodios/243 000 decisiones.

## Registro

- Inicio: v7 no existía; rama `feature/ddqn-experiments` con cambios previos
  en v5, v6, .gitignore y protocolo v6. Se preservan.
- Comprobaciones locales limitadas a JSON/AST/compilación; sin datasets,
  modelos, simulación ni smoke local. Contratos numéricos se ejecutan dentro
  del run FULL remoto antes de entrenar.
- Estado remoto, interno, calibración e integridad se reportan por separado.
- Notebook v7 creado desde el v6 local actual. Red, replay, codificación continua
  y make_replay conservados mediante comparación AST. No se ejecutaron celdas
  experimentales localmente.
- Contratos numéricos y reconstrucción real de A-test se ejecutan en Kaggle
  antes de entrenar DDQN. No se considera corregido el fallo histórico de F1
  hasta observar esos contratos remotos aprobados.
- Revisión estática adicional terminada. Corregido el rollback de cachés:
  solo se reutilizan activos cuya publicación está confirmada en RUN_STATE.
- Enviado FULL T4, timeout 43200 s, versión 1; Kaggle generó el slug desde el
  título inicial: `jeffreyamc/darl-ddqn-v7-full-auc-ap` (id 140267881).
  Última consulta: RUNNING. No se lanzó otra ejecución para corregir el nombre.
  Título local corregido para futuros envíos; conservar/reutilizar este trabajo.
- Pendiente: contratos remotos, seguimiento, consolidación con identidad
  prevista y auditoría del barrido. No hay ganador ni outputs descargados.

## Persistencia y continuaciones

`prepared_replays.pkl.gz`, `prepared_initial.pkl.gz` y
`prepared_calibration.pkl.gz` conservan los activos completos, incluido
`Pipeline.ref`. Los pickle se cargan solo desde outputs privados propios.
`resume.pt` vincula el agente y todos sus generadores a offsets de CSV append
y copias exactas de tablas reemplazables. Una continuación restaura esos
límites antes de seguir; no duplica filas de unidades incompletas.

Cada diez episodios se guardan online/target, Adam, buffer, RNG privado del
buffer, RNG del agente, RNG globales, epsilon, learning rate, actualizaciones,
historia y ventanas de selección. Baselines y test usan unidades independientes
por escenario/política o escenario/configuración. La verificación guarda hashes
de los archivos ya comprobados. Presupuesto de diez horas; timeout solicitado
de doce horas. `partial` requiere continuación y nunca acredita 1000 episodios.

## Interpretación

Ganador: mejor media de tres validaciones consecutivas de utilidad común,
cuatro escenarios con igual peso. Se conserva el último checkpoint de la
ventana. Desempate por AP, AUC, menor tiempo e identificador. Test no interviene.
Curvas nativas de rewards diferentes no establecen superioridad. La banda de
una semilla es temporal, no incertidumbre entre semillas. AUC/AP/costo y
calibración se reportan separados, sin exigir mejoría para aprobar integridad.


## Resultados verificados — 27 de octubre de 2026

Ejecución real `jeffreyamc/darl-ddqn-v7-full-auc-ap`, versión 1: COMPLETE remoto, complete interno, 9000 episodios/243 000 decisiones, 74 comprobaciones aprobadas. Nueve configuraciones completas; no se requieren continuaciones. Calibración no cualificada. Seleccionada α=0.75, β=0.5, λ=0.025, checkpoint 890 por B-validation. Descargados 439 outputs (245 CSV, 155 PNG; 1.89 GB). Gráficas legibles y hallazgos en kaggle/dqn-dummy/v7/outputs/analysis-20261027/README.md. No se ejecutó entrenamiento adicional ni local.
