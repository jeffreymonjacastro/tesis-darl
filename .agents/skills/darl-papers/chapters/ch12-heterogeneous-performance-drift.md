# Who Experiences Large Model Decay and Why: A Hierarchical Framework for Diagnosing Heterogeneous Performance Drift

## Core Idea

**Role in DARL:** subgroup and slice-level performance drift diagnosis.

Highlights that model decay may not affect all groups equally. For DARL this matters because an overall AUC drop can hide subgroup-specific degradation, and a maintenance policy trained only on aggregate metrics may miss clinically or operationally important drift.

This chapter is intentionally written as an actionable research note rather than a paper summary. Use it when you need to decide whether this paper supports a claim in the thesis, which part of DARL it informs, and which claims should be kept separate.

## Research Problem

The paper addresses the gap captured by its role: subgroup and slice-level performance drift diagnosis. In the DARL thesis this matters because selective updating is only defensible when the literature supports three separable ideas:

- deployed tabular systems can degrade under changing data conditions;
- diagnosis and action selection are different steps;
- maintenance should consider cost, severity, and uncertainty instead of defaulting to full retraining.

For this source, the strongest thesis use is: Heterogeneous performance drift, subgroup decay diagnosis, slice-level model monitoring..

## Method or Framework

The paper offers a hierarchical diagnostic perspective for identifying who experiences decay and why. In DARL it can motivate future extensions where the observation vector includes subgroup drift, slice performance, or fairness-relevant diagnostics, even if the initial implementation focuses on aggregate metrics.

When applying the source, preserve the distinction between the paper's original method and the DARL framework. The source can motivate or justify a component, but DARL's contribution remains the integration of diagnostics, selective actions, and cost-sensitive sequential decision-making for a two-stage tabular pipeline.

## Datasets, Metrics, or Evaluation Signals

Subgroup performance, heterogeneous model decay, slice diagnostics, group-specific degradation.

For the DARL experimental chapter, translate these signals into observable quantities only when the implementation can actually compute them. Do not cite this paper as evidence for a metric that is absent from the experiment logs.

## How DARL Uses This Paper

Use it when discussing limitations: selective updating should eventually account for who is affected by drift, not only how much the global metric moves. It also supports adding subgroup diagnostics to future state representations.

Recommended thesis placement:

- **Introduction / Motivation:** use when it explains why model maintenance matters in deployed tabular ML.
- **Related Work:** use when comparing DARL against drift diagnosis, cost-aware retraining, self-healing pipelines, or RL-based control.
- **Methodology:** use only when the paper directly supports state variables, action design, reward terms, or algorithm choice.
- **Limitations / Future Work:** use when the paper motivates subgroup drift, broader monitoring, or richer adaptation policies not yet implemented.

## What To Cite It For

Heterogeneous performance drift, subgroup decay diagnosis, slice-level model monitoring.

## Do Not Cite It For

Current PPO objective, TableShift setup unless subgroup splits are used, or XGBoost training details.

## Limitations and Guardrails

This paper should not be overused if the current DARL experiments do not implement subgroup-aware actions. It is strongest as related work and future-work motivation.

Do not copy the paper's wording into the thesis. Use this chapter to build your own Spanish academic explanation, then cite the original source in LaTeX/BibTeX.

## Key Concepts

- model decay
- heterogeneous drift
- subgroup diagnostics
- slice performance
- fairness risk

## Connects To

- Diagnosing Model Performance Under Distribution Shift
- Impact of HbA1c Measurement on Hospital Readmission

## Extraction Notes

- Source file: `Who experiences large model decay and why A Hierarchical Framework for Diagnosing Heterogeneous Performance Drift.pdf`
- Extractor: `pdftotext`
- Pages: 31
- Words extracted: 18743
- Skill chapter: `ch12-heterogeneous-performance-drift.md`
