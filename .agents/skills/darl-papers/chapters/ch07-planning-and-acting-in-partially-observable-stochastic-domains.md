# Planning and Acting in Partially Observable Stochastic Domains

## Core Idea

**Role in DARL:** formal foundation for treating drift maintenance as partial observability.

Supports the claim that DARL is naturally a partially observable decision problem. The true cause of model degradation is not directly observed; the agent observes noisy diagnostics, delayed performance effects, and incomplete evidence about whether a stage update will help.

This chapter is intentionally written as an actionable research note rather than a paper summary. Use it when you need to decide whether this paper supports a claim in the thesis, which part of DARL it informs, and which claims should be kept separate.

## Research Problem

The paper addresses the gap captured by its role: formal foundation for treating drift maintenance as partial observability. In the DARL thesis this matters because selective updating is only defensible when the literature supports three separable ideas:

- deployed tabular systems can degrade under changing data conditions;
- diagnosis and action selection are different steps;
- maintenance should consider cost, severity, and uncertainty instead of defaulting to full retraining.

For this source, the strongest thesis use is: POMDP formulation, belief states, partial observability, stochastic decision domains..

## Method or Framework

The paper develops planning and acting under stochastic partial observability. In DARL this maps to a hidden state representing the actual data-generating condition and pipeline health, observations from drift detectors and metrics, actions that update stages, and rewards that combine performance recovery with costs.

When applying the source, preserve the distinction between the paper's original method and the DARL framework. The source can motivate or justify a component, but DARL's contribution remains the integration of diagnostics, selective actions, and cost-sensitive sequential decision-making for a two-stage tabular pipeline.

## Datasets, Metrics, or Evaluation Signals

Belief-state quality, expected utility, planning under uncertainty, action value under partial information.

For the DARL experimental chapter, translate these signals into observable quantities only when the implementation can actually compute them. Do not cite this paper as evidence for a metric that is absent from the experiment logs.

## How DARL Uses This Paper

Use this paper in Chapter III to justify a POMDP framing: the agent does not know the exact drift mechanism, only an observation vector. This makes selective updating a sequential decision under uncertainty, not a deterministic rule table.

Recommended thesis placement:

- **Introduction / Motivation:** use when it explains why model maintenance matters in deployed tabular ML.
- **Related Work:** use when comparing DARL against drift diagnosis, cost-aware retraining, self-healing pipelines, or RL-based control.
- **Methodology:** use only when the paper directly supports state variables, action design, reward terms, or algorithm choice.
- **Limitations / Future Work:** use when the paper motivates subgroup drift, broader monitoring, or richer adaptation policies not yet implemented.

## What To Cite It For

POMDP formulation, belief states, partial observability, stochastic decision domains.

## Do Not Cite It For

PPO clipping, GAE, TableShift, or XGBoost.

## Limitations and Guardrails

Classical POMDP planning can be computationally difficult and does not prescribe PPO. Use it for the problem formulation, while using modern RL papers for the practical learning algorithm.

Do not copy the paper's wording into the thesis. Use this chapter to build your own Spanish academic explanation, then cite the original source in LaTeX/BibTeX.

## Key Concepts

- POMDP
- partial observability
- belief state
- stochastic domain
- expected utility

## Connects To

- Proximal Policy Optimization Algorithms
- High-dimensional continuous control using generalized advantage estimation
- `rl-book` skill for RL textbook foundations

## Extraction Notes

- Source file: `Planning and acting in partially observable stochastic domains.pdf`
- Extractor: `pdftotext`
- Pages: 36
- Words extracted: 16434
- Skill chapter: `ch07-planning-and-acting-in-partially-observable-stochastic-domains.md`
