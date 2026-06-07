# Survey on Concept Drift Adaptation

## Core Idea

**Role in DARL:** taxonomy and vocabulary for concept drift and adaptation.

Provides the definitional backbone for concept drift: what changes, how drift appears over time, and what families of adaptation methods exist. DARL uses this survey to avoid vague language and to distinguish covariate shift from changes in the conditional relationship between inputs and labels.

This chapter is intentionally written as an actionable research note rather than a paper summary. Use it when you need to decide whether this paper supports a claim in the thesis, which part of DARL it informs, and which claims should be kept separate.

## Research Problem

The paper addresses the gap captured by its role: taxonomy and vocabulary for concept drift and adaptation. In the DARL thesis this matters because selective updating is only defensible when the literature supports three separable ideas:

- deployed tabular systems can degrade under changing data conditions;
- diagnosis and action selection are different steps;
- maintenance should consider cost, severity, and uncertainty instead of defaulting to full retraining.

For this source, the strongest thesis use is: Concept drift definitions, taxonomy, adaptation strategies, online drift vocabulary..

## Method or Framework

As a survey, its method is classification and synthesis rather than one new algorithm. It organizes drift types, detection approaches, adaptation strategies, and evaluation concerns. This is useful in the related work chapter and when defining the synthetic drift injector.

When applying the source, preserve the distinction between the paper's original method and the DARL framework. The source can motivate or justify a component, but DARL's contribution remains the integration of diagnostics, selective actions, and cost-sensitive sequential decision-making for a two-stage tabular pipeline.

## Datasets, Metrics, or Evaluation Signals

Concept drift definitions, detection categories, adaptation strategies, streaming evaluation terms.

For the DARL experimental chapter, translate these signals into observable quantities only when the implementation can actually compute them. Do not cite this paper as evidence for a metric that is absent from the experiment logs.

## How DARL Uses This Paper

Use it to define concept drift, abrupt versus gradual or recurring changes, and the need for adaptation. It also helps explain why label-flip synthetic drift is only a controlled proxy for changes in P(Y|X).

Recommended thesis placement:

- **Introduction / Motivation:** use when it explains why model maintenance matters in deployed tabular ML.
- **Related Work:** use when comparing DARL against drift diagnosis, cost-aware retraining, self-healing pipelines, or RL-based control.
- **Methodology:** use only when the paper directly supports state variables, action design, reward terms, or algorithm choice.
- **Limitations / Future Work:** use when the paper motivates subgroup drift, broader monitoring, or richer adaptation policies not yet implemented.

## What To Cite It For

Concept drift definitions, taxonomy, adaptation strategies, online drift vocabulary.

## Do Not Cite It For

Cost-aware RL reward design, TableShift, PPO, or XGBoost internals.

## Limitations and Guardrails

A survey provides breadth, not a direct experiment for DARL. Cite it for definitions and taxonomy, then use specific empirical papers for benchmark and algorithmic claims.

Do not copy the paper's wording into the thesis. Use this chapter to build your own Spanish academic explanation, then cite the original source in LaTeX/BibTeX.

## Key Concepts

- concept drift
- adaptation
- streaming learning
- drift taxonomy
- abrupt drift

## Connects To

- Diagnosing Model Performance Under Distribution Shift
- Severity-Aware Drift Adaptation
- Self-Healing ML Pipelines

## Extraction Notes

- Source file: `Survey on Concept Drift Adaptation.pdf`
- Extractor: `pdftotext`
- Pages: 37
- Words extracted: 22278
- Skill chapter: `ch11-survey-on-concept-drift-adaptation.md`
