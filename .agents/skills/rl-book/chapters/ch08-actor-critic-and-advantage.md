# Actor-Critic and Advantage

## Core Idea

Actor-critic methods combine a policy model, the actor, with a value model, the critic. The critic estimates value so the actor can update using lower-variance advantage signals instead of raw returns. The advantage asks whether an action was better or worse than expected for the current state.

## DARL Interpretation

Actor-critic is a natural fit for PPO-based DARL. The actor selects the maintenance action, while the critic estimates expected utility of the current drift state. Advantage signals can reward selective updates that recover performance at lower cost and penalize unnecessary full retraining.

## Formula Anchor

```text
A_pi(s,a) = Q_pi(s,a) - V_pi(s)
```

Advantage compares the value of a specific action with the average value of the state under the policy.

Use the thesis formula style when moving this into LaTeX: equation, `Donde:`, symbol explanation, then conceptual explanation in Spanish.

## Writing Guidance

- Keep the explanation academic and concise.
- Tie every RL term to the two-stage pipeline maintenance setting.
- Avoid presenting RL as the thesis contribution; the contribution is the DARL framing and empirical evaluation.
- Use this chapter together with `patterns.md` when drafting methodology text.

## Key Terms

- actor
- critic
- advantage
- variance reduction
- PPO connection

## Extraction Notes

- Source file: `RL An Introduction.pdf`
- Extractor: `pdftotext`
- Source pages: 548
- Source words: 261233
- Skill chapter: `ch08-actor-critic-and-advantage.md`
