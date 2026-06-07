# Mapping RL Concepts to DARL Design

## Core Idea

The RL book should be used as a conceptual foundation, not as a source for drift-specific claims. The DARL environment maps RL components into the maintenance domain: observations are drift and performance signals; actions are selective update choices; rewards combine predictive recovery and resource cost; episodes simulate a stream before and after drift.

## DARL Interpretation

When writing Chapter III, explicitly define the RL elements in DARL terms. Avoid generic RL exposition that does not connect to the implemented environment. The reader should see how return, discounting, and policy optimization correspond to update decisions over time.

## Formula Anchor

```text
r_t = performance_gain_t - lambda_cost cost(a_t)
```

This reward sketch captures DARL's central tradeoff: recover performance while penalizing expensive maintenance actions.

Use the thesis formula style when moving this into LaTeX: equation, `Donde:`, symbol explanation, then conceptual explanation in Spanish.

## Writing Guidance

- Keep the explanation academic and concise.
- Tie every RL term to the two-stage pipeline maintenance setting.
- Avoid presenting RL as the thesis contribution; the contribution is the DARL framing and empirical evaluation.
- Use this chapter together with `patterns.md` when drafting methodology text.

## Key Terms

- DARL state
- action
- reward
- episode
- baselines
- thesis wording

## Extraction Notes

- Source file: `RL An Introduction.pdf`
- Extractor: `pdftotext`
- Source pages: 548
- Source words: 261233
- Skill chapter: `ch10-darl-rl-design.md`
