# Dynamic Programming

## Core Idea

Dynamic programming solves planning problems when a complete environment model is known. Policy evaluation estimates the value of a fixed policy; policy improvement makes the policy greedy with respect to value estimates; value iteration combines evaluation and improvement. These methods are important conceptually even when they are not practical for DARL.

## DARL Interpretation

DARL usually does not know exact transition probabilities for drift and maintenance outcomes. Still, dynamic programming provides the clean theoretical baseline: if the transition and reward model were known, one could compute a policy. The implemented RL approach is a sample-based alternative because the environment is simulated from datasets, drift injection, and update costs.

## Formula Anchor

```text
v_{k+1}(s) = max_a E[R_{t+1} + gamma v_k(S_{t+1}) | S_t=s, A_t=a]
```

Value iteration updates state values toward the best action's expected return.

Use the thesis formula style when moving this into LaTeX: equation, `Donde:`, symbol explanation, then conceptual explanation in Spanish.

## Writing Guidance

- Keep the explanation academic and concise.
- Tie every RL term to the two-stage pipeline maintenance setting.
- Avoid presenting RL as the thesis contribution; the contribution is the DARL framing and empirical evaluation.
- Use this chapter together with `patterns.md` when drafting methodology text.

## Key Terms

- policy evaluation
- policy improvement
- value iteration
- known models

## Extraction Notes

- Source file: `RL An Introduction.pdf`
- Extractor: `pdftotext`
- Source pages: 548
- Source words: 261233
- Skill chapter: `ch04-dynamic-programming.md`
