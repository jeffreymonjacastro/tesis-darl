# Glossary

- **Covariate shift:** change in the input distribution P(X). In DARL this can motivate preprocessing-stage updates or feature-distribution diagnostics.
- **Concept drift:** change in P(Y|X). In DARL this can motivate predictive-model updates and label/error-stream monitoring.
- **Performance decay:** degradation of predictive metrics over time or across shifted environments.
- **Selective updating:** maintenance action that updates only the necessary stage instead of retraining the whole pipeline.
- **Cost-aware retraining:** retraining decision process that accounts for compute, time, memory, and operational cost.
- **Severity-aware adaptation:** adaptation strategy whose response intensity depends on drift magnitude or performance loss.
- **Self-healing pipeline:** MLOps loop that monitors, diagnoses, remediates, and evaluates pipeline health.
- **POMDP:** decision model where the true system state is hidden and the agent acts from observations or beliefs.
- **PPO:** policy-gradient method using a clipped surrogate objective to stabilize policy updates.
- **GAE:** generalized advantage estimation; a bias-variance controlled estimator for actor-critic training.
- **TableShift:** benchmark framework for tabular distribution shift evaluation.
- **XGBoost:** regularized gradient tree boosting system used as a strong tabular predictive model.
- **Heterogeneous drift:** performance decay that affects subgroups or slices differently.
- **Action cost:** penalty associated with maintenance actions, such as retraining time or peak RAM.
- **Observation vector:** DARL state input containing drift, performance, cost, and episode-context signals.
