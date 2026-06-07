# High-dimensional Continuous Control Using Generalized Advantage Estimation

## Core Idea

**Role in DARL:** policy-gradient variance reduction reference for PPO-style training.

Introduces generalized advantage estimation as a practical way to trade bias and variance in policy gradient methods. DARL can use it to explain why an actor-critic policy can learn from episodic maintenance trajectories without relying on raw high-variance returns alone.

This chapter is intentionally written as an actionable research note rather than a paper summary. Use it when you need to decide whether this paper supports a claim in the thesis, which part of DARL it informs, and which claims should be kept separate.

## Research Problem

The paper addresses the gap captured by its role: policy-gradient variance reduction reference for PPO-style training. In the DARL thesis this matters because selective updating is only defensible when the literature supports three separable ideas:

- deployed tabular systems can degrade under changing data conditions;
- diagnosis and action selection are different steps;
- maintenance should consider cost, severity, and uncertainty instead of defaulting to full retraining.

For this source, the strongest thesis use is: GAE, advantage estimation, actor-critic training, bias-variance tradeoff in policy gradients..

## Method or Framework

GAE computes an advantage signal by accumulating temporal-difference residuals with a discount factor and a trace parameter. The trace parameter controls the balance between short-horizon bootstrapped estimates and longer-horizon Monte Carlo-like returns. This is relevant when the reward for an update decision is delayed because its effect appears over future stream windows.

When applying the source, preserve the distinction between the paper's original method and the DARL framework. The source can motivate or justify a component, but DARL's contribution remains the integration of diagnostics, selective actions, and cost-sensitive sequential decision-making for a two-stage tabular pipeline.

## Datasets, Metrics, or Evaluation Signals

Advantage estimates, policy-gradient learning stability, variance reduction, episodic return.

For the DARL experimental chapter, translate these signals into observable quantities only when the implementation can actually compute them. Do not cite this paper as evidence for a metric that is absent from the experiment logs.

## How DARL Uses This Paper

Use it in the methodology when explaining the training objective of the RL policy. It connects the DARL reward sequence to a learnable advantage signal: update actions that improve future utility should receive positive advantage even if immediate cost is nonzero.

Recommended thesis placement:

- **Introduction / Motivation:** use when it explains why model maintenance matters in deployed tabular ML.
- **Related Work:** use when comparing DARL against drift diagnosis, cost-aware retraining, self-healing pipelines, or RL-based control.
- **Methodology:** use only when the paper directly supports state variables, action design, reward terms, or algorithm choice.
- **Limitations / Future Work:** use when the paper motivates subgroup drift, broader monitoring, or richer adaptation policies not yet implemented.

## What To Cite It For

GAE, advantage estimation, actor-critic training, bias-variance tradeoff in policy gradients.

## Do Not Cite It For

Tabular drift benchmarks, concept drift adaptation taxonomy, or MLOps automation.

## Limitations and Guardrails

The paper is not about dataset drift or model maintenance. It should only support the RL optimization machinery, especially if the implementation uses PPO or an actor-critic method with advantage estimates.

Do not copy the paper's wording into the thesis. Use this chapter to build your own Spanish academic explanation, then cite the original source in LaTeX/BibTeX.

## Key Concepts

- generalized advantage estimation
- advantage
- TD residual
- policy gradient
- actor-critic

## Connects To

- Proximal Policy Optimization Algorithms
- `rl-book` skill for RL textbook foundations

## Extraction Notes

- Source file: `High-dimensional continuous control using generalized advantage estimation.pdf`
- Extractor: `pdftotext`
- Pages: 14
- Words extracted: 7796
- Skill chapter: `ch04-generalized-advantage-estimation.md`
