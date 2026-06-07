# Patterns

## Related Work Pattern

Use one paragraph per function:

1. Benchmark papers show the empirical reality of tabular distribution shift.
2. Drift diagnosis papers explain why degradation must be interpreted before acting.
3. Cost-aware and severity-aware papers justify selective maintenance rather than full retraining by default.
4. RL/POMDP/PPO papers justify sequential decision-making under uncertainty.
5. XGBoost and healthcare-domain papers justify the pipeline and dataset choices.

## Citation Guardrail Pattern

When writing a thesis claim, check:

- Is the paper evidence for the phenomenon, method, metric, or only motivation?
- Does DARL implement the same mechanism, or merely use the paper as conceptual support?
- Is the claim about detection, diagnosis, adaptation, or policy learning?

## DARL Integration Pattern

For each source, map it into one of these roles:

- **Environment evidence:** TableShift, healthcare readmission.
- **Diagnostic evidence:** distribution shift diagnosis, concept drift survey, heterogeneous decay.
- **Action-cost evidence:** cost-aware retraining, resource allocation, severity-aware adaptation.
- **RL/control evidence:** POMDP, PPO, GAE.
- **Pipeline evidence:** XGBoost, self-healing ML pipelines.

## Anti-Patterns

- Do not cite PPO or GAE as evidence for drift detection.
- Do not cite TableShift as an adaptation method.
- Do not cite XGBoost as a self-healing system.
- Do not cite healthcare readmission context as proof that DARL works.
- Do not imply subgroup-aware maintenance if the current implementation only evaluates aggregate metrics.
