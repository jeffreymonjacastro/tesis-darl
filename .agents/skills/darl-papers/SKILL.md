---
name: darl-papers
description: "Knowledge base from DARL thesis papers excluding the RL book. Use for literature review, TableShift, distribution shift diagnosis, concept drift, cost-aware retraining, self-healing pipelines, PPO, POMDP support papers, XGBoost, and connecting papers to the DARL thesis argument."
---

# DARL Papers

Use this skill when working with the academic papers that support the DARL thesis, excluding Sutton and Barto's RL book. The skill follows the `book-to-skill` pattern: keep this `SKILL.md` as a compact router and load only the chapter file needed for the current paper or topic.

## Source Scope

- Sources: 13 PDFs from `literature/papers/`.
- RL textbook source is intentionally handled by the separate `rl-book` skill.
- Extracted with: `multi-method` in `text` mode.
- Approximate source size: 318 pages, 170903 words, ~227K tokens.
- Generated: 2026-06-07.

## How To Use

1. Identify whether the user asks for a paper, concept, thesis section, or implementation decision.
2. Load only the matching chapter in `chapters/`.
3. Use `glossary.md` for terminology, `patterns.md` for citation/argument templates, and `cheatsheet.md` for quick routing.
4. Do not quote long passages from PDFs. Write thesis text in your own Spanish academic prose and cite the original paper through BibTeX.

## Chapter Router

- `ch01-benchmarking-distribution-shift-in-tabular-data.md` - TableShift benchmark: benchmark and evaluation protocol for tabular distribution shift.
- `ch02-cost-aware-retraining-for-machine-learning.md` - cost-aware retraining: maintenance baseline for deciding when retraining is worth its cost.
- `ch03-diagnosing-model-performance-under-distribution-shift.md` - shift diagnosis: diagnostic foundation for separating performance decay from observed distribution changes.
- `ch04-generalized-advantage-estimation.md` - GAE: policy-gradient variance reduction reference for PPO-style training.
- `ch05-impact-of-hba1c-measurement-on-hospital-readmission.md` - readmission domain: healthcare/domain context for hospital readmission prediction.
- `ch06-optimal-resource-allocation-for-ml-model-training-and-maintenance.md` - resource allocation: resource-budget motivation for selective updating.
- `ch07-planning-and-acting-in-partially-observable-stochastic-domains.md` - POMDP foundation: formal foundation for treating drift maintenance as partial observability.
- `ch08-proximal-policy-optimization-algorithms.md` - PPO: practical policy optimization method for the DARL agent.
- `ch09-self-healing-ml-pipelines.md` - self-healing MLOps: MLOps motivation for automated detection and remediation.
- `ch10-severity-aware-drift-adaptation-for-cost-efficient-model-maintenance.md` - severity-aware adaptation: severity-conditioned maintenance policy motivation.
- `ch11-survey-on-concept-drift-adaptation.md` - concept drift survey: taxonomy and vocabulary for concept drift and adaptation.
- `ch12-heterogeneous-performance-drift.md` - heterogeneous decay: subgroup and slice-level performance drift diagnosis.
- `ch13-xgboost-a-scalable-tree-boosting-system.md` - XGBoost: stage-2 predictive model foundation for tabular pipelines.

## Topic Router

- **TableShift / tabular OOD:** read `chapters/ch01-benchmarking-distribution-shift-in-tabular-data.md`.
- **Cost-aware maintenance:** read `chapters/ch02-cost-aware-retraining-for-machine-learning.md`, `chapters/ch06-optimal-resource-allocation-for-ml-model-training-and-maintenance.md`, and `chapters/ch10-severity-aware-drift-adaptation-for-cost-efficient-model-maintenance.md`.
- **Drift diagnosis:** read `chapters/ch03-diagnosing-model-performance-under-distribution-shift.md` and `chapters/ch12-heterogeneous-performance-drift.md`.
- **Concept drift taxonomy:** read `chapters/ch11-survey-on-concept-drift-adaptation.md`.
- **Self-healing MLOps:** read `chapters/ch09-self-healing-ml-pipelines.md`.
- **POMDP / PPO support papers:** read `chapters/ch07-planning-and-acting-in-partially-observable-stochastic-domains.md`, `chapters/ch08-proximal-policy-optimization-algorithms.md`, and `chapters/ch04-generalized-advantage-estimation.md`.
- **Healthcare/readmission context:** read `chapters/ch05-impact-of-hba1c-measurement-on-hospital-readmission.md`.
- **Predictive model stage:** read `chapters/ch13-xgboost-a-scalable-tree-boosting-system.md`.

## DARL Argument Map

- **Problem:** deployed tabular pipelines degrade under distribution shift and concept drift.
- **Observation:** drift statistics, error streams, and performance deltas give partial evidence about hidden causes.
- **Decision:** update no stage, preprocessing, predictive model, or the full pipeline depending on diagnosis, severity, and cost.
- **Learning:** PPO/POMDP support papers justify sequential decision-making under uncertainty.
- **Evaluation:** TableShift, XGBoost, cost-aware maintenance, and self-healing papers justify benchmarks, model stage, metrics, and baselines.

## Guardrails

- Do not treat all papers as evidence for all claims; each chapter lists what the paper should and should not support.
- Do not merge this with `rl-book`; the RL textbook is deliberately separate to avoid loading a large RL reference when the user asks for one paper.
- Do not write summaries into `literature/notes/`; this skill is the reusable knowledge layer.
