# Partial Observability

## Core Idea

Partial observability occurs when the agent cannot directly observe the full environment state. The agent receives observations that may be noisy, incomplete, or delayed. A belief state or history can help summarize uncertainty about the hidden state.

## DARL Interpretation

DARL's true drift cause is hidden: observed PSI, KS, ADWIN flags, and AUC drops are evidence, not the cause itself. This makes a POMDP framing more accurate than a fully observed MDP. The thesis should state that the implementation uses observable diagnostics as a practical approximation of belief.

## Formula Anchor

```text
b_t(s) = Pr(S_t=s | O_1, A_1, ..., O_t)
```

A belief state is a probability distribution over hidden states conditioned on observation and action history.

Use the thesis formula style when moving this into LaTeX: equation, `Donde:`, symbol explanation, then conceptual explanation in Spanish.

## Writing Guidance

- Keep the explanation academic and concise.
- Tie every RL term to the two-stage pipeline maintenance setting.
- Avoid presenting RL as the thesis contribution; the contribution is the DARL framing and empirical evaluation.
- Use this chapter together with `patterns.md` when drafting methodology text.

## Key Terms

- hidden state
- observations
- belief
- history
- POMDP connection

## Extraction Notes

- Source file: `RL An Introduction.pdf`
- Extractor: `pdftotext`
- Source pages: 548
- Source words: 261233
- Skill chapter: `ch09-partial-observability.md`
