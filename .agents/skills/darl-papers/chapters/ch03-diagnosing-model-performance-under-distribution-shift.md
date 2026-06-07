# Diagnosing Model Performance Under Distribution Shift

## Core Idea

**Role in DARL:** diagnostic foundation for separating performance decay from observed distribution changes.

Supports the diagnostic side of DARL: performance degradation is not enough information for a good maintenance action. The agent needs observations that distinguish what changed, how severe the change is, and whether the likely cause is closer to covariate shift, concept drift, subgroup drift, or a combination.

This chapter is intentionally written as an actionable research note rather than a paper summary. Use it when you need to decide whether this paper supports a claim in the thesis, which part of DARL it informs, and which claims should be kept separate.

## Research Problem

The paper addresses the gap captured by its role: diagnostic foundation for separating performance decay from observed distribution changes. In the DARL thesis this matters because selective updating is only defensible when the literature supports three separable ideas:

- deployed tabular systems can degrade under changing data conditions;
- diagnosis and action selection are different steps;
- maintenance should consider cost, severity, and uncertainty instead of defaulting to full retraining.

For this source, the strongest thesis use is: Distribution-shift diagnosis, performance decay analysis, covariate/concept evidence, diagnostic observations..

## Method or Framework

The paper emphasizes diagnostic decomposition under distribution shift. In DARL terms this motivates a state representation that includes drift statistics, performance deltas, subgroup indicators, and possibly error-stream signals. It also supports separating detection from action: detecting drift is not the same as knowing which part of the pipeline to refit.

When applying the source, preserve the distinction between the paper's original method and the DARL framework. The source can motivate or justify a component, but DARL's contribution remains the integration of diagnostics, selective actions, and cost-sensitive sequential decision-making for a two-stage tabular pipeline.

## Datasets, Metrics, or Evaluation Signals

Performance degradation, distributional discrepancy, subgroup or slice-level changes, diagnostic indicators.

For the DARL experimental chapter, translate these signals into observable quantities only when the implementation can actually compute them. Do not cite this paper as evidence for a metric that is absent from the experiment logs.

## How DARL Uses This Paper

Use this paper when describing the observation vector: PSI/KS-style feature drift, AUC drop, error stream behavior, and diagnostic signals should be available to the policy. It also strengthens the claim that selective updating is a diagnosis-conditioned decision.

Recommended thesis placement:

- **Introduction / Motivation:** use when it explains why model maintenance matters in deployed tabular ML.
- **Related Work:** use when comparing DARL against drift diagnosis, cost-aware retraining, self-healing pipelines, or RL-based control.
- **Methodology:** use only when the paper directly supports state variables, action design, reward terms, or algorithm choice.
- **Limitations / Future Work:** use when the paper motivates subgroup drift, broader monitoring, or richer adaptation policies not yet implemented.

## What To Cite It For

Distribution-shift diagnosis, performance decay analysis, covariate/concept evidence, diagnostic observations.

## Do Not Cite It For

PPO algorithm details, TableShift dataset construction, or XGBoost model design.

## Limitations and Guardrails

Diagnostic methods can explain or localize degradation, but do not automatically optimize a sequence of maintenance actions under compute cost. They are evidence sources for the DARL state, not a direct replacement for the RL decision policy.

Do not copy the paper's wording into the thesis. Use this chapter to build your own Spanish academic explanation, then cite the original source in LaTeX/BibTeX.

## Key Concepts

- shift diagnosis
- performance degradation
- covariate shift
- concept drift
- diagnostic state

## Connects To

- Benchmarking Distribution Shift in Tabular Data
- Who experiences large model decay and why
- Survey on Concept Drift Adaptation

## Extraction Notes

- Source file: `Diagnosing Model Performance Under Distribution Shift.pdf`
- Extractor: `pdftotext`
- Pages: 41
- Words extracted: 18797
- Skill chapter: `ch03-diagnosing-model-performance-under-distribution-shift.md`
