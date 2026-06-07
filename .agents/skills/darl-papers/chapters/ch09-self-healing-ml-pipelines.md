# Self-Healing ML Pipelines

## Core Idea

**Role in DARL:** MLOps motivation for automated detection and remediation.

Frames model maintenance as an operational pipeline problem: a deployed ML system should monitor, diagnose, and remediate degradation with minimal manual intervention. This is close to DARL's practical motivation, where the agent acts as a maintenance controller for a two-stage pipeline.

This chapter is intentionally written as an actionable research note rather than a paper summary. Use it when you need to decide whether this paper supports a claim in the thesis, which part of DARL it informs, and which claims should be kept separate.

## Research Problem

The paper addresses the gap captured by its role: MLOps motivation for automated detection and remediation. In the DARL thesis this matters because selective updating is only defensible when the literature supports three separable ideas:

- deployed tabular systems can degrade under changing data conditions;
- diagnosis and action selection are different steps;
- maintenance should consider cost, severity, and uncertainty instead of defaulting to full retraining.

For this source, the strongest thesis use is: Self-healing ML, MLOps remediation loops, automated maintenance, monitor-diagnose-act framing..

## Method or Framework

The paper discusses self-healing mechanisms such as monitoring, triggering, adaptation, and feedback. In DARL these become a closed loop: observe drift and performance, choose an update action, apply it, measure the result, and learn from the episode.

When applying the source, preserve the distinction between the paper's original method and the DARL framework. The source can motivate or justify a component, but DARL's contribution remains the integration of diagnostics, selective actions, and cost-sensitive sequential decision-making for a two-stage tabular pipeline.

## Datasets, Metrics, or Evaluation Signals

Operational recovery, drift detection, intervention cost, post-remediation performance, automation quality.

For the DARL experimental chapter, translate these signals into observable quantities only when the implementation can actually compute them. Do not cite this paper as evidence for a metric that is absent from the experiment logs.

## How DARL Uses This Paper

Use it in the introduction and related work to position DARL as a self-healing maintenance approach with explicit action choices. It also supports the need for monitoring outputs to feed an automated controller rather than only a dashboard.

Recommended thesis placement:

- **Introduction / Motivation:** use when it explains why model maintenance matters in deployed tabular ML.
- **Related Work:** use when comparing DARL against drift diagnosis, cost-aware retraining, self-healing pipelines, or RL-based control.
- **Methodology:** use only when the paper directly supports state variables, action design, reward terms, or algorithm choice.
- **Limitations / Future Work:** use when the paper motivates subgroup drift, broader monitoring, or richer adaptation policies not yet implemented.

## What To Cite It For

Self-healing ML, MLOps remediation loops, automated maintenance, monitor-diagnose-act framing.

## Do Not Cite It For

PPO mathematics, XGBoost internals, or TableShift split definitions.

## Limitations and Guardrails

Self-healing terminology can be broad. It may describe architectures and principles rather than a specific empirical policy for selective stage refitting. Avoid citing it as proof that DARL's policy outperforms baselines.

Do not copy the paper's wording into the thesis. Use this chapter to build your own Spanish academic explanation, then cite the original source in LaTeX/BibTeX.

## Key Concepts

- self-healing ML
- MLOps
- monitoring
- remediation
- closed-loop maintenance

## Connects To

- Cost-aware retraining for machine learning
- Diagnosing Model Performance Under Distribution Shift
- Severity-Aware Drift Adaptation

## Extraction Notes

- Source file: `Self-Healing ML Pipelines.pdf`
- Extractor: `pdftotext`
- Pages: 13
- Words extracted: 5156
- Skill chapter: `ch09-self-healing-ml-pipelines.md`
