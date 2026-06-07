# XGBoost: A Scalable Tree Boosting System

## Core Idea

**Role in DARL:** stage-2 predictive model foundation for tabular pipelines.

Provides the model-side foundation for using XGBoost as the main predictive learner in a tabular pipeline. It supports the thesis choice of a strong, widely used baseline whose performance can still decay under distribution shift.

This chapter is intentionally written as an actionable research note rather than a paper summary. Use it when you need to decide whether this paper supports a claim in the thesis, which part of DARL it informs, and which claims should be kept separate.

## Research Problem

The paper addresses the gap captured by its role: stage-2 predictive model foundation for tabular pipelines. In the DARL thesis this matters because selective updating is only defensible when the literature supports three separable ideas:

- deployed tabular systems can degrade under changing data conditions;
- diagnosis and action selection are different steps;
- maintenance should consider cost, severity, and uncertainty instead of defaulting to full retraining.

For this source, the strongest thesis use is: Gradient tree boosting, regularized objective, sparse feature handling, tabular baseline model..

## Method or Framework

XGBoost combines gradient tree boosting with regularization, second-order optimization, sparsity-aware split handling, approximate split finding, and systems-level scalability. In DARL it belongs to stage 2 of the pipeline: after preprocessing produces features, XGBoost learns the predictive mapping.

When applying the source, preserve the distinction between the paper's original method and the DARL framework. The source can motivate or justify a component, but DARL's contribution remains the integration of diagnostics, selective actions, and cost-sensitive sequential decision-making for a two-stage tabular pipeline.

## Datasets, Metrics, or Evaluation Signals

Predictive performance, training scalability, regularized objective, tree boosting efficiency.

For the DARL experimental chapter, translate these signals into observable quantities only when the implementation can actually compute them. Do not cite this paper as evidence for a metric that is absent from the experiment logs.

## How DARL Uses This Paper

Use it to explain why the predictive model stage is independently refittable and why a model update action has different meaning from a preprocessing update. A model-only update can respond to changes in P(Y|X) without necessarily rebuilding the feature transformation stage.

Recommended thesis placement:

- **Introduction / Motivation:** use when it explains why model maintenance matters in deployed tabular ML.
- **Related Work:** use when comparing DARL against drift diagnosis, cost-aware retraining, self-healing pipelines, or RL-based control.
- **Methodology:** use only when the paper directly supports state variables, action design, reward terms, or algorithm choice.
- **Limitations / Future Work:** use when the paper motivates subgroup drift, broader monitoring, or richer adaptation policies not yet implemented.

## What To Cite It For

Gradient tree boosting, regularized objective, sparse feature handling, tabular baseline model.

## Do Not Cite It For

Drift detection, PPO, POMDP, or cost-aware maintenance.

## Limitations and Guardrails

XGBoost is a powerful predictor, not a drift-maintenance policy. It should be cited to justify the model stage and baseline strength, while drift diagnosis and update policy are handled by other work.

Do not copy the paper's wording into the thesis. Use this chapter to build your own Spanish academic explanation, then cite the original source in LaTeX/BibTeX.

## Key Concepts

- XGBoost
- gradient tree boosting
- regularization
- sparsity-aware split
- tabular ML

## Connects To

- Benchmarking Distribution Shift in Tabular Data
- Cost-aware retraining for machine learning

## Extraction Notes

- Source file: `XGBoost.pdf`
- Extractor: `pdftotext`
- Pages: 10
- Words extracted: 8457
- Skill chapter: `ch13-xgboost-a-scalable-tree-boosting-system.md`
