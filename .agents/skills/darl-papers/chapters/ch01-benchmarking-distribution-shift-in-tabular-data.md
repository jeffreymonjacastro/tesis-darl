# Benchmarking Distribution Shift in Tabular Data

## Core Idea

**Role in DARL:** benchmark and evaluation protocol for tabular distribution shift.

Provides the empirical motivation for treating tabular distribution shift as a first-class evaluation problem rather than as an afterthought of ordinary train/test validation. For DARL, the paper is most useful as a bridge between real-world dataset shift and a controlled experimental protocol where a reference environment and shifted environments can be compared with consistent metrics.

This chapter is intentionally written as an actionable research note rather than a paper summary. Use it when you need to decide whether this paper supports a claim in the thesis, which part of DARL it informs, and which claims should be kept separate.

## Research Problem

The paper addresses the gap captured by its role: benchmark and evaluation protocol for tabular distribution shift. In the DARL thesis this matters because selective updating is only defensible when the literature supports three separable ideas:

- deployed tabular systems can degrade under changing data conditions;
- diagnosis and action selection are different steps;
- maintenance should consider cost, severity, and uncertainty instead of defaulting to full retraining.

For this source, the strongest thesis use is: TableShift, tabular shift benchmarks, OOD tabular evaluation, reference/post-drift split design..

## Method or Framework

The work organizes tabular prediction tasks into benchmark scenarios with explicit distribution differences between source and target splits. Its methodological value is not a new learner, but a repeatable evaluation surface: fixed datasets, defined splits, baseline models, and performance under shift. In the thesis this supports the choice to separate pre-drift and post-drift evaluation windows instead of reporting a single aggregate test metric.

When applying the source, preserve the distinction between the paper's original method and the DARL framework. The source can motivate or justify a component, but DARL's contribution remains the integration of diagnostics, selective actions, and cost-sensitive sequential decision-making for a two-stage tabular pipeline.

## Datasets, Metrics, or Evaluation Signals

AUC, accuracy-type predictive metrics, in-distribution versus out-of-distribution performance comparisons, dataset-level shift descriptors.

For the DARL experimental chapter, translate these signals into observable quantities only when the implementation can actually compute them. Do not cite this paper as evidence for a metric that is absent from the experiment logs.

## How DARL Uses This Paper

Use this paper to justify TableShift-style data selection, realistic tabular shift settings, and the need to evaluate post-drift behavior. In the DARL argument it anchors the environment: the RL agent acts because tabular deployments face measurable distribution shifts that ordinary static validation does not capture.

Recommended thesis placement:

- **Introduction / Motivation:** use when it explains why model maintenance matters in deployed tabular ML.
- **Related Work:** use when comparing DARL against drift diagnosis, cost-aware retraining, self-healing pipelines, or RL-based control.
- **Methodology:** use only when the paper directly supports state variables, action design, reward terms, or algorithm choice.
- **Limitations / Future Work:** use when the paper motivates subgroup drift, broader monitoring, or richer adaptation policies not yet implemented.

## What To Cite It For

TableShift, tabular shift benchmarks, OOD tabular evaluation, reference/post-drift split design.

## Do Not Cite It For

RL policy design, PPO, stage-specific update decisions, or formal concept drift taxonomy.

## Limitations and Guardrails

The benchmark does not by itself solve diagnosis or adaptation. It can show that performance degrades under shift, but it does not decide whether to update preprocessing, the predictive model, or both. It should therefore be cited for experimental grounding, not as a selective maintenance policy.

Do not copy the paper's wording into the thesis. Use this chapter to build your own Spanish academic explanation, then cite the original source in LaTeX/BibTeX.

## Key Concepts

- TableShift
- distribution shift
- tabular benchmarks
- OOD evaluation
- reference environment

## Connects To

- Diagnosing Model Performance Under Distribution Shift
- Self-Healing ML Pipelines
- XGBoost

## Extraction Notes

- Source file: `Benchmarking Distribution Shift in Tabular Data.pdf`
- Extractor: `pdftotext`
- Pages: 48
- Words extracted: 25280
- Skill chapter: `ch01-benchmarking-distribution-shift-in-tabular-data.md`
