# Severity-Aware Drift Adaptation for Cost-Efficient Model Maintenance

## Core Idea

**Role in DARL:** severity-conditioned maintenance policy motivation.

Strengthens DARL's premise that drift response should depend on severity. A small shift may not justify a full retrain; a severe or persistent shift may. This supports making severity an input to the state and a factor in reward shaping or action thresholds.

This chapter is intentionally written as an actionable research note rather than a paper summary. Use it when you need to decide whether this paper supports a claim in the thesis, which part of DARL it informs, and which claims should be kept separate.

## Research Problem

The paper addresses the gap captured by its role: severity-conditioned maintenance policy motivation. In the DARL thesis this matters because selective updating is only defensible when the literature supports three separable ideas:

- deployed tabular systems can degrade under changing data conditions;
- diagnosis and action selection are different steps;
- maintenance should consider cost, severity, and uncertainty instead of defaulting to full retraining.

For this source, the strongest thesis use is: Severity-aware drift adaptation, drift magnitude, cost-efficient maintenance, action intensity..

## Method or Framework

The paper links drift severity to adaptation intensity and cost efficiency. In DARL, severity can be computed from detector outputs, performance deltas, or synthetic drift parameters, then exposed to the agent so it can learn differentiated actions.

When applying the source, preserve the distinction between the paper's original method and the DARL framework. The source can motivate or justify a component, but DARL's contribution remains the integration of diagnostics, selective actions, and cost-sensitive sequential decision-making for a two-stage tabular pipeline.

## Datasets, Metrics, or Evaluation Signals

Drift severity, adaptation cost, performance recovery, cost-efficiency, maintenance action intensity.

For the DARL experimental chapter, translate these signals into observable quantities only when the implementation can actually compute them. Do not cite this paper as evidence for a metric that is absent from the experiment logs.

## How DARL Uses This Paper

Use it to justify the `severity` dimension in experiments and the idea that action choice should scale with drift magnitude. It also supports reporting cost-efficient recovery rather than only maximum post-update AUC.

Recommended thesis placement:

- **Introduction / Motivation:** use when it explains why model maintenance matters in deployed tabular ML.
- **Related Work:** use when comparing DARL against drift diagnosis, cost-aware retraining, self-healing pipelines, or RL-based control.
- **Methodology:** use only when the paper directly supports state variables, action design, reward terms, or algorithm choice.
- **Limitations / Future Work:** use when the paper motivates subgroup drift, broader monitoring, or richer adaptation policies not yet implemented.

## What To Cite It For

Severity-aware drift adaptation, drift magnitude, cost-efficient maintenance, action intensity.

## Do Not Cite It For

POMDP theory, PPO clipping, or clinical readmission facts.

## Limitations and Guardrails

Severity-aware adaptation may still be rule-based or model-level rather than stage-specific. It should be used to justify severity-sensitive decisions, not as a complete replacement for the RL formulation.

Do not copy the paper's wording into the thesis. Use this chapter to build your own Spanish academic explanation, then cite the original source in LaTeX/BibTeX.

## Key Concepts

- drift severity
- adaptation intensity
- cost efficiency
- maintenance policy
- performance recovery

## Connects To

- Cost-aware retraining for machine learning
- Optimal Resource Allocation for ML Model Training and
- Survey on Concept Drift Adaptation

## Extraction Notes

- Source file: `Severity-Aware Drift Adaptation for Cost-Efficient Model Maintenance.pdf`
- Extractor: `pdftotext`
- Pages: 24
- Words extracted: 10518
- Skill chapter: `ch10-severity-aware-drift-adaptation-for-cost-efficient-model-maintenance.md`
