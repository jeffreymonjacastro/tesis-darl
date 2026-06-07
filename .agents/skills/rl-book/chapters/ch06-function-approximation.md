# Function Approximation

## Core Idea

Function approximation replaces tabular value estimates with parameterized functions. This is required when the state space is large, continuous, or built from many diagnostic features. Rather than storing one value per state, the learner maps features to values or policies through a model such as a linear approximator or neural network.

## DARL Interpretation

DARL's observation vector can include numeric drift statistics, AUC drops, detector flags, action history, and cost signals. This is not a small tabular state space. Function approximation justifies using a policy network or actor-critic architecture that generalizes across drift severities, datasets, and maintenance histories.

## Formula Anchor

```text
v_hat(s, w) approx v_pi(s)
```

The parameter vector w lets the value approximation generalize across states with similar features.

Use the thesis formula style when moving this into LaTeX: equation, `Donde:`, symbol explanation, then conceptual explanation in Spanish.

## Writing Guidance

- Keep the explanation academic and concise.
- Tie every RL term to the two-stage pipeline maintenance setting.
- Avoid presenting RL as the thesis contribution; the contribution is the DARL framing and empirical evaluation.
- Use this chapter together with `patterns.md` when drafting methodology text.

## Key Terms

- generalization
- parameterized value functions
- features
- neural approximators

## Extraction Notes

- Source file: `RL An Introduction.pdf`
- Extractor: `pdftotext`
- Source pages: 548
- Source words: 261233
- Skill chapter: `ch06-function-approximation.md`
