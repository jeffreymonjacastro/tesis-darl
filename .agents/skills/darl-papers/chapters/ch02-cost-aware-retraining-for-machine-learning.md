# Cost-aware Retraining for Machine Learning

## Core Idea

**Role in DARL:** maintenance baseline for deciding when retraining is worth its cost.

Frames retraining as an economic decision rather than a purely predictive one. The central idea for DARL is that a model update has an opportunity cost: time, compute, memory, operational risk, and possibly delayed service. A useful maintenance policy must therefore compare expected performance recovery against the cost of the action.

This chapter is intentionally written as an actionable research note rather than a paper summary. Use it when you need to decide whether this paper supports a claim in the thesis, which part of DARL it informs, and which claims should be kept separate.

## Research Problem

The paper addresses the gap captured by its role: maintenance baseline for deciding when retraining is worth its cost. In the DARL thesis this matters because selective updating is only defensible when the literature supports three separable ideas:

- deployed tabular systems can degrade under changing data conditions;
- diagnosis and action selection are different steps;
- maintenance should consider cost, severity, and uncertainty instead of defaulting to full retraining.

For this source, the strongest thesis use is: Cost-aware model maintenance, retraining cost, utility-cost tradeoffs, reward penalty motivation..

## Method or Framework

The paper studies retraining under explicit cost constraints and encourages decision criteria that balance degradation and resource usage. For DARL this maps naturally to the reward function: a full pipeline update should not be optimal merely because it maximizes AUC if a cheaper stage-specific update recovers most of the loss.

When applying the source, preserve the distinction between the paper's original method and the DARL framework. The source can motivate or justify a component, but DARL's contribution remains the integration of diagnostics, selective actions, and cost-sensitive sequential decision-making for a two-stage tabular pipeline.

## Datasets, Metrics, or Evaluation Signals

Predictive utility, retraining cost, compute/resource cost, policy value under maintenance decisions.

For the DARL experimental chapter, translate these signals into observable quantities only when the implementation can actually compute them. Do not cite this paper as evidence for a metric that is absent from the experiment logs.

## How DARL Uses This Paper

Use it to justify reward penalties for expensive actions, the inclusion of retraining time and RAM in metrics, and the comparison against full retraining. The paper helps argue that DARL is not merely seeking accuracy recovery; it is seeking cost-aware recovery.

Recommended thesis placement:

- **Introduction / Motivation:** use when it explains why model maintenance matters in deployed tabular ML.
- **Related Work:** use when comparing DARL against drift diagnosis, cost-aware retraining, self-healing pipelines, or RL-based control.
- **Methodology:** use only when the paper directly supports state variables, action design, reward terms, or algorithm choice.
- **Limitations / Future Work:** use when the paper motivates subgroup drift, broader monitoring, or richer adaptation policies not yet implemented.

## What To Cite It For

Cost-aware model maintenance, retraining cost, utility-cost tradeoffs, reward penalty motivation.

## Do Not Cite It For

POMDP formulation, drift diagnosis, or evidence that preprocessing-only updates are sufficient.

## Limitations and Guardrails

Cost-aware retraining is usually framed at the model-maintenance level, not at the two-stage pipeline level. It may not distinguish whether degradation comes from feature distribution changes, label mechanism changes, preprocessing mismatch, or model decay.

Do not copy the paper's wording into the thesis. Use this chapter to build your own Spanish academic explanation, then cite the original source in LaTeX/BibTeX.

## Key Concepts

- cost-aware retraining
- maintenance utility
- compute budget
- resource-aware ML
- retraining policy

## Connects To

- Optimal Resource Allocation for ML Model Training and
- Severity-Aware Drift Adaptation
- Self-Healing ML Pipelines

## Extraction Notes

- Source file: `Cost-aware retraining for machine learning.pdf`
- Extractor: `pdftotext`
- Pages: 16
- Words extracted: 12891
- Skill chapter: `ch02-cost-aware-retraining-for-machine-learning.md`
