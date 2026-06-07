# Monte Carlo and Temporal-Difference Learning

## Core Idea

Monte Carlo methods learn from complete sampled returns, while temporal-difference methods learn from one-step or multi-step bootstrapped targets. TD learning is central to RL because it updates estimates from experience without waiting for perfect models. The TD error measures the surprise between a predicted value and a reward-plus-next-value target.

## DARL Interpretation

DARL episodes can be treated as sampled maintenance trajectories. TD-style reasoning explains how the agent can learn from partial trajectories: an update action's quality is assessed through immediate cost and future performance, not only final episode score. This is also useful when explaining actor-critic methods and advantage estimates.

## Formula Anchor

```text
delta_t = R_{t+1} + gamma V(S_{t+1}) - V(S_t)
```

The TD error delta_t is the gap between the bootstrapped target and the current value estimate.

Use the thesis formula style when moving this into LaTeX: equation, `Donde:`, symbol explanation, then conceptual explanation in Spanish.

## Writing Guidance

- Keep the explanation academic and concise.
- Tie every RL term to the two-stage pipeline maintenance setting.
- Avoid presenting RL as the thesis contribution; the contribution is the DARL framing and empirical evaluation.
- Use this chapter together with `patterns.md` when drafting methodology text.

## Key Terms

- Monte Carlo return
- TD error
- bootstrapping
- online learning

## Extraction Notes

- Source file: `RL An Introduction.pdf`
- Extractor: `pdftotext`
- Source pages: 548
- Source words: 261233
- Skill chapter: `ch05-monte-carlo-and-temporal-difference-learning.md`
